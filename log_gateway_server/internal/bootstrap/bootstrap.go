// Package bootstrap implements the interactive startup guidance of the log
// gateway. Before the service comes up it:
//
//  1. prints a banner,
//  2. resolves and probes the configured Python interpreter (agent.python),
//  3. detects whether the scienceflow package is importable in that
//     environment and which version is installed,
//  4. queries the package index for the latest release (with a --pre
//     fallback, because scienceflow only publishes pre-releases),
//  5. offers to pip-install / pip-upgrade automatically after user
//     confirmation (declining a required install aborts startup with the
//     manual command shown),
//  6. prints a summary of the effective gateway configuration.
//
// Non-interactive runs (stdin not a TTY, e.g. nohup/systemd) never hang: a
// missing scienceflow install aborts with the manual command, a pending
// upgrade is skipped with a notice. Use the -yes flag for unattended
// install/upgrade runs.
package bootstrap

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"log_gateway_server/internal/config"
)

// Options controls bootstrap behaviour.
type Options struct {
	// AssumeYes answers every confirmation prompt with "yes" (-yes flag):
	// missing packages are installed and pending upgrades applied without
	// asking, which enables unattended runs.
	AssumeYes bool
}

// ErrAborted is returned when the bootstrap cannot continue and the gateway
// must not start: interpreter unavailable, a required scienceflow install was
// declined (or failed). main() exits non-zero on this error.
var ErrAborted = errors.New("bootstrap aborted")

// Run executes the full startup guidance described in the package comment.
// All user-facing output goes to stdout; prompts read from stdin. A nil
// return means the gateway may proceed with normal startup.
func Run(cfg *config.Config, opts Options) error {
	u := &ui{w: os.Stdout, color: useColor()}
	stdin := bufio.NewReader(os.Stdin)
	interactive := stdinInteractive()

	ag := cfg.Agent
	dist := distOf(ag.Module)

	u.blank()
	u.banner()
	u.quiet("启动自检：Python 环境 · scienceflow 安装与版本更新 · 配置总览")

	// ---- [1/3] Python interpreter -------------------------------------
	u.step(1, 3, "检测 Python 环境")
	raw := splitCommand(ag.Python)
	if len(raw) == 0 {
		u.fail("agent.python 未配置")
		return fmt.Errorf("%w: agent.python is empty", ErrAborted)
	}
	base := prepareBase(raw)
	exe, pyVer, pyErr := pythonInfo(base)
	if pyErr != nil {
		u.fail("无法运行解释器 %q：%v", ag.Python, pyErr)
		if ag.Enabled {
			return fmt.Errorf("%w: interpreter %q unavailable: %v", ErrAborted, ag.Python, pyErr)
		}
		u.info("agent 未启用，跳过环境引导，继续启动")
		printSummary(u, cfg, "", "", "")
		return nil
	}
	u.ok("Python %s · %s", pyVer, exe)

	// ---- [2/3] scienceflow installation -------------------------------
	u.step(2, 3, "检测 scienceflow 安装")
	cur, curErr := sfVersion(base, dist)
	installed := curErr == nil
	if installed {
		u.ok("已安装 scienceflow %s", cur)
	} else {
		u.warn("未检测到 scienceflow（包名 %s）", dist)
	}

	// ---- [3/3] latest version on the index -----------------------------
	u.step(3, 3, "检查最新版本")
	latest, pre, latErr := latestVersion(base, dist)
	if latErr != nil {
		u.warn("无法获取索引上的最新版本（网络或 pip 索引不可达）")
		u.quiet("%v", latErr)
	} else {
		u.ok("索引最新版本 %s", latest)
	}

	// ---- decision: install / upgrade / up-to-date ----------------------
	if !installed {
		if !ag.Enabled {
			u.info("agent.enabled=false，scienceflow 非必需，跳过安装引导")
		} else {
			v, err := ensureInstalled(u, stdin, interactive, opts, raw, base, dist, pre, exe, pyVer)
			if err != nil {
				return err
			}
			cur = v
			u.ok("安装完成：scienceflow %s", v)
		}
	} else if latErr == nil {
		c, cmpOK := cmpVer(cur, latest)
		switch {
		case cmpOK && c < 0:
			maybeUpgrade(u, stdin, interactive, opts, raw, base, dist, pre, cur, latest, ag.Enabled)
		case cmpOK:
			u.ok("已是最新版本（%s）", cur)
		default:
			u.quiet("无法比较版本 %s 与 %s，保留当前版本", cur, latest)
		}
	} else {
		u.quiet("保留当前版本 %s", cur)
	}

	// ---- config summary -------------------------------------------------
	printSummary(u, cfg, cur, exe, pyVer)

	u.blank()
	u.quiet("引导完成，正在启动网关服务…")
	return nil
}

// ensureInstalled handles the "scienceflow missing" branch. It shows the
// Python path and the exact install command, asks for confirmation and either
// runs pip install (streaming output) or aborts startup per the requirement.
// On success it returns the verified installed version.
func ensureInstalled(u *ui, stdin *bufio.Reader, interactive bool, opts Options,
	raw, base []string, dist string, pre bool, exe, pyVer string) (string, error) {

	insArgs := pipArgs(raw, dist, false, pre)
	cmdline := strings.Join(insArgs, " ")

	u.blank()
	u.warn("需要安装 scienceflow 才能运行 agent 任务")
	u.kv("Python", fmt.Sprintf("%s (%s)", exe, pyVer))
	u.kv("安装命令", cmdline)

	approved := opts.AssumeYes
	if !approved {
		if !interactive {
			u.blank()
			u.fail("非交互环境无法确认安装。请手动执行上方命令后重新启动网关")
			return "", fmt.Errorf("%w: scienceflow missing (non-interactive)", ErrAborted)
		}
		approved = u.confirm("是否自动安装 scienceflow？", true, stdin)
	}
	if !approved {
		u.blank()
		u.fail("已取消安装。请手动执行以下命令后重新启动：")
		u.print("      %s", u.paint(cBold, cmdline))
		return "", fmt.Errorf("%w: install declined", ErrAborted)
	}

	u.blank()
	if err := runStreaming(pipArgs(base, dist, false, pre), u.w, u.color); err != nil {
		u.fail("安装失败：%v", err)
		u.quiet("请手动执行：%s", cmdline)
		return "", fmt.Errorf("%w: pip install failed", ErrAborted)
	}
	v, err := sfVersion(base, dist)
	if err != nil {
		u.fail("安装后仍未检测到 scienceflow，请检查 pip 环境")
		return "", fmt.Errorf("%w: install verification failed", ErrAborted)
	}
	return v, nil
}

// maybeUpgrade handles the "newer version available" branch: confirmation
// prompt, pip install --upgrade with streamed output, and post-upgrade
// verification. All failure paths keep the current version and continue.
func maybeUpgrade(u *ui, stdin *bufio.Reader, interactive bool, opts Options,
	raw, base []string, dist string, pre bool, cur, latest string, agentEnabled bool) {

	upArgs := pipArgs(raw, dist, true, pre)
	cmdline := strings.Join(upArgs, " ")

	u.warn("发现新版本：%s（当前 %s）", latest, cur)
	if !agentEnabled {
		u.info("agent 未启用，跳过更新")
		return
	}

	approved := opts.AssumeYes
	if !approved {
		if !interactive {
			u.info("非交互环境，跳过自动更新（保留当前版本）")
			u.quiet("手动更新命令：%s", cmdline)
			return
		}
		u.blank()
		approved = u.confirm("是否自动更新 scienceflow？", true, stdin)
	}
	if !approved {
		u.info("保留当前版本 %s", cur)
		return
	}

	u.blank()
	if err := runStreaming(pipArgs(base, dist, true, pre), u.w, u.color); err != nil {
		u.fail("更新失败：%v", err)
		u.quiet("保留当前版本 %s 继续启动", cur)
		return
	}
	if v, e := sfVersion(base, dist); e == nil {
		u.ok("更新完成：scienceflow %s", v)
	} else {
		u.warn("更新后版本校验失败，保留当前版本 %s", cur)
	}
}

// ---------------------------------------------------------------------------
// Environment probes
// ---------------------------------------------------------------------------

const probeScript = "import sys;print(sys.executable);v=sys.version_info;print('%d.%d.%d'%(v[0],v[1],v[2]))"

// pythonInfo resolves the real interpreter behind base and its X.Y.Z version.
func pythonInfo(base []string) (exe, version string, err error) {
	out, _, err := runCapture(base, []string{"-c", probeScript}, 30*time.Second)
	if err != nil {
		return "", "", err
	}
	lines := nonEmptyLines(out)
	if len(lines) < 2 {
		return "", "", fmt.Errorf("unexpected interpreter probe output")
	}
	return lines[0], lines[1], nil
}

var errNotInstalled = errors.New("package metadata not found")

// sfVersion returns the installed distribution version via
// importlib.metadata; errNotInstalled when the package is absent.
func sfVersion(base []string, dist string) (string, error) {
	script := fmt.Sprintf("from importlib.metadata import version;print(version(%q))", dist)
	out, _, err := runCapture(base, []string{"-c", script}, 30*time.Second)
	if err != nil {
		return "", errNotInstalled
	}
	v := strings.TrimSpace(out)
	if v == "" {
		return "", errNotInstalled
	}
	return nonEmptyLines(v)[0], nil
}

// latestVersion queries `<py> -m pip index versions <dist>`. The plain query
// runs first; if the index has no stable releases (scienceflow publishes
// pre-releases only) it is retried with --pre, and `pre` reports that the
// subsequent install/upgrade needs the same flag.
func latestVersion(base []string, dist string) (latest string, pre bool, err error) {
	var lastErr error
	for _, withPre := range []bool{false, true} {
		extra := []string{"-m", "pip", "index", "versions", dist}
		if withPre {
			extra = append(extra, "--pre")
		}
		out, errOut, e := runCapture(base, extra, 30*time.Second)
		if e != nil {
			lastErr = fmt.Errorf("%v: %s", e, lastLine(errOut))
			continue
		}
		if v := parsePipIndexVersions(out, dist); v != "" {
			return v, withPre, nil
		}
	}
	if lastErr == nil {
		lastErr = errors.New("index returned no versions")
	}
	return "", false, lastErr
}

// pipArgs builds the full pip command line for install/upgrade. base is the
// prepared interpreter argv ("python", or "uv run --no-sync python", ...).
func pipArgs(base []string, dist string, upgrade, pre bool) []string {
	args := append([]string{}, base...)
	args = append(args, "-m", "pip", "install")
	if upgrade {
		args = append(args, "--upgrade")
	}
	if pre {
		args = append(args, "--pre")
	}
	return append(args, dist)
}

// prepareBase mirrors the agent runner's uv handling: `uv run python` gets
// --no-sync so probes never trigger a network dependency resolution.
func prepareBase(argv []string) []string {
	if len(argv) > 0 && argv[0] == "uv" && containsToken(argv, "run") &&
		!containsToken(argv, "--no-sync") && !containsToken(argv, "--frozen") {
		return insertAfter(argv, "run", "--no-sync")
	}
	return argv
}

// runCapture runs argv+extra capturing stdout/stderr separately, with a
// hard timeout (killed on expiry).
func runCapture(argv, extra []string, timeout time.Duration) (string, string, error) {
	args := append(append([]string{}, argv...), extra...)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	var out, errOut bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errOut
	err := cmd.Run()
	return out.String(), errOut.String(), err
}

// runStreaming runs args, piping stdout+stderr to w with a dim "│ " indent so
// pip output nests visually under the banner flow.
func runStreaming(args []string, w io.Writer, color bool) error {
	prefix := "  │ "
	if color {
		prefix = cDim + prefix + cReset
	}
	cmd := exec.Command(args[0], args[1:]...)
	iw := &indentWriter{w: w, prefix: prefix, atStart: true}
	cmd.Stdout, cmd.Stderr = iw, iw
	return cmd.Run()
}

// indentWriter prefixes every line with prefix. Safe for concurrent use (it
// is attached to both subprocess stdout and stderr).
type indentWriter struct {
	mu      sync.Mutex
	w       io.Writer
	prefix  string
	atStart bool
}

func (iw *indentWriter) Write(p []byte) (int, error) {
	iw.mu.Lock()
	defer iw.mu.Unlock()
	written := 0
	for len(p) > 0 {
		if iw.atStart {
			if _, err := io.WriteString(iw.w, iw.prefix); err != nil {
				return written, err
			}
			iw.atStart = false
		}
		end := len(p)
		hasNL := false
		if i := bytes.IndexByte(p, '\n'); i >= 0 {
			end = i + 1
			hasNL = true
		}
		n, err := iw.w.Write(p[:end])
		written += n
		if err != nil {
			return written, err
		}
		if n < end {
			// Partial write: resume mid-line without re-indenting.
			p = p[n:]
			continue
		}
		if hasNL {
			iw.atStart = true
		}
		p = p[end:]
	}
	return written, nil
}

// ---------------------------------------------------------------------------
// Version parsing / comparison (PEP 440 subset)
// ---------------------------------------------------------------------------

type parsedVersion struct {
	nums   []int
	rank   int // 0 dev, 1 alpha, 2 beta, 3 rc, 4 final, 5 post
	preNum int
}

var preRank = map[string]int{
	"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2,
	"c": 3, "rc": 3, "pre": 3, "": 4, "post": 5,
}

var reVersion = regexp.MustCompile(`^v?(\d+(?:\.\d+)*)(?:[._-]?(dev|a|alpha|b|beta|c|rc|pre|post)[._-]?(\d*))?$`)

func parseVersion(s string) (parsedVersion, bool) {
	m := reVersion.FindStringSubmatch(strings.ToLower(strings.TrimSpace(s)))
	if m == nil {
		return parsedVersion{}, false
	}
	v := parsedVersion{rank: preRank[m[2]]}
	for _, part := range strings.Split(m[1], ".") {
		n, err := strconv.Atoi(part)
		if err != nil {
			return parsedVersion{}, false
		}
		v.nums = append(v.nums, n)
	}
	if m[3] != "" {
		v.preNum, _ = strconv.Atoi(m[3])
	}
	return v, true
}

// cmpVer returns -1 / 0 / +1 when a orders before / equal / after b, and
// ok=false when either side is not a recognisable version.
func cmpVer(a, b string) (int, bool) {
	pa, oka := parseVersion(a)
	pb, okb := parseVersion(b)
	if !oka || !okb {
		return 0, false
	}
	n := len(pa.nums)
	if len(pb.nums) > n {
		n = len(pb.nums)
	}
	for i := 0; i < n; i++ {
		x, y := 0, 0
		if i < len(pa.nums) {
			x = pa.nums[i]
		}
		if i < len(pb.nums) {
			y = pb.nums[i]
		}
		if x != y {
			return sign(x - y), true
		}
	}
	if pa.rank != pb.rank {
		return sign(pa.rank - pb.rank), true
	}
	return sign(pa.preNum - pb.preNum), true
}

func sign(x int) int {
	switch {
	case x < 0:
		return -1
	case x > 0:
		return 1
	}
	return 0
}

var (
	reIndexPkg   = regexp.MustCompile(`(?im)^\s*([A-Za-z0-9._-]+) \(([^)]+)\)`)
	reIndexAvail = regexp.MustCompile(`(?im)^Available versions:\s*(.+)$`)
)

// parsePipIndexVersions extracts the newest version from `pip index versions`
// output. Primary form is the "name (1.2.3)" line; fallback is the first item
// of the "Available versions:" list (pip prints it newest-first).
func parsePipIndexVersions(out, dist string) string {
	for _, m := range reIndexPkg.FindAllStringSubmatch(out, -1) {
		if strings.EqualFold(m[1], dist) {
			return strings.TrimSpace(m[2])
		}
	}
	if m := reIndexAvail.FindStringSubmatch(out); m != nil {
		for _, part := range strings.Split(m[1], ",") {
			if p := strings.TrimSpace(part); p != "" {
				return p
			}
		}
	}
	return ""
}

// distOf maps the CLI module ("scienceflow.cli") to its distribution name
// ("scienceflow").
func distOf(module string) string {
	if i := strings.IndexByte(module, '.'); i > 0 {
		return module[:i]
	}
	return module
}

// ---------------------------------------------------------------------------
// argv helpers (mirror of the agent runner's splitCommand)
// ---------------------------------------------------------------------------

// splitCommand splits "uv run python" style strings into argv tokens,
// honouring double quotes for paths containing spaces.
func splitCommand(s string) []string {
	var tokens []string
	var cur strings.Builder
	inQuote := false
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '"':
			inQuote = !inQuote
		case (c == ' ' || c == '\t') && !inQuote:
			flush()
		default:
			cur.WriteByte(c)
		}
	}
	flush()
	return tokens
}

func containsToken(argv []string, token string) bool {
	for _, a := range argv {
		if a == token {
			return true
		}
	}
	return false
}

func insertAfter(argv []string, anchor, token string) []string {
	out := make([]string, 0, len(argv)+1)
	for _, a := range argv {
		out = append(out, a)
		if a == anchor {
			out = append(out, token)
			anchor = "\x00" // only insert once
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Terminal UI
// ---------------------------------------------------------------------------

const (
	cReset = "\x1b[0m"
	cBold  = "\x1b[1m"
	cDim   = "\x1b[2m"
	cRed   = "\x1b[31m"
	cGreen = "\x1b[32m"
	cYel   = "\x1b[33m"
	cCyan  = "\x1b[36m"
)

type ui struct {
	w     io.Writer
	color bool
}

func (u *ui) paint(color, s string) string {
	if !u.color || color == "" {
		return s
	}
	return color + s + cReset
}

func (u *ui) print(format string, a ...any) {
	fmt.Fprintf(u.w, format+"\n", a...)
}

func (u *ui) blank() {
	fmt.Fprintln(u.w)
}

func (u *ui) banner() {
	const width = 66
	title := "ScienceFlow Log Gateway · 启动引导"
	sub := "环境自检 · 版本更新 · 配置总览"
	inner := width - 2
	u.print("%s", u.paint(cCyan, "┌"+strings.Repeat("─", inner)+"┐"))
	u.print("%s%s%s", u.paint(cCyan, "│"), center(u.paint(cBold, title), inner), u.paint(cCyan, "│"))
	u.print("%s%s%s", u.paint(cCyan, "│"), center(u.paint(cDim, sub), inner), u.paint(cCyan, "│"))
	u.print("%s", u.paint(cCyan, "└"+strings.Repeat("─", inner)+"┘"))
}

// center pads s (ANSI-aware: padding computed on the plain-text width) to
// display width w, with one leading space.
func center(s string, w int) string {
	pad := w - 1 - dispWidth(stripANSI(s))
	if pad < 0 {
		pad = 0
	}
	return " " + s + strings.Repeat(" ", pad)
}

func (u *ui) rule(title string) {
	const width = 66
	plain := " " + title + " "
	fill := width - 1 - dispWidth(plain)
	if fill < 0 {
		fill = 0
	}
	u.print("%s%s%s",
		u.paint(cDim, "─"),
		u.paint(cBold+cCyan, plain),
		u.paint(cDim, strings.Repeat("─", fill)))
}

func (u *ui) step(i, n int, title string) {
	u.blank()
	u.print("%s %s %s",
		u.paint(cCyan, "◆"),
		u.paint(cDim, fmt.Sprintf("[%d/%d]", i, n)),
		u.paint(cBold, title))
}

func (u *ui) ok(format string, a ...any) {
	u.print("  %s %s", u.paint(cGreen, "✓"), fmt.Sprintf(format, a...))
}

func (u *ui) warn(format string, a ...any) {
	u.print("  %s %s", u.paint(cYel, "⚠"), u.paint(cYel, fmt.Sprintf(format, a...)))
}

func (u *ui) fail(format string, a ...any) {
	u.print("  %s %s", u.paint(cRed, "✗"), u.paint(cRed+cBold, fmt.Sprintf(format, a...)))
}

func (u *ui) info(format string, a ...any) {
	u.print("  %s %s", u.paint(cCyan, "ℹ"), fmt.Sprintf(format, a...))
}

func (u *ui) quiet(format string, a ...any) {
	u.print("  %s", u.paint(cDim, fmt.Sprintf(format, a...)))
}

// kv prints a "label value" summary row; the label is dimmed and padded to a
// fixed display width so values align (CJK-aware).
func (u *ui) kv(label, value string) {
	const col = 14
	pad := col - dispWidth(label)
	if pad < 1 {
		pad = 1
	}
	u.print("  %s%s%s", u.paint(cDim, label), strings.Repeat(" ", pad), value)
}

// confirm prints a yes/no prompt and reads the answer. Empty input picks
// def; EOF is treated as "no" (never silently installs unattended).
func (u *ui) confirm(question string, def bool, stdin *bufio.Reader) bool {
	hint := "y/N"
	if def {
		hint = "Y/n"
	}
	for tries := 0; tries < 3; tries++ {
		fmt.Fprintf(u.w, "  %s %s %s ", u.paint(cBold+cCyan, "?"), question, u.paint(cDim, "["+hint+"]"))
		line, err := stdin.ReadString('\n')
		if err != nil {
			fmt.Fprintln(u.w)
			return false
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "":
			return def
		case "y", "yes":
			return true
		case "n", "no":
			return false
		}
		u.print("  %s 无效输入，请输入 y 或 n", u.paint(cYel, "⚠"))
	}
	return def
}

// printSummary renders the effective configuration panel shown after the
// environment guidance, right before the gateway starts.
func printSummary(u *ui, cfg *config.Config, sfVer, exe, pyVer string) {
	ag := cfg.Agent
	u.blank()
	u.rule("网关配置")
	u.kv("服务地址", fmt.Sprintf("http://%s:%d", cfg.Server.Host, cfg.Server.Port))
	u.kv("SSE 端点", cfg.Server.SSEPath)
	if cfg.Auth.Enabled {
		u.kv("鉴权", fmt.Sprintf("已启用 · %s · token %s", cfg.Auth.File, formatDur(cfg.Auth.TokenTTL.Duration)))
	} else {
		u.kv("鉴权", u.paint(cYel, "未启用"))
	}
	u.kv("后端反代", cfg.Backend.URL)
	u.kv("日志源", fmt.Sprintf("%d 个 · 扫描 %s", len(cfg.Inputs), formatDur(cfg.Scan.Frequency.Duration)))
	u.kv("检查点", cfg.Registry.Path)
	if ag.Enabled {
		if sfVer != "" {
			u.kv("scienceflow", u.paint(cGreen, sfVer))
		} else {
			u.kv("scienceflow", u.paint(cDim, "未安装"))
		}
		if exe != "" {
			u.kv("Python", fmt.Sprintf("%s (%s)", exe, pyVer))
		}
		u.kv("模块/命令", fmt.Sprintf("%s · %s", ag.Module, ag.Command))
		mode := "site-packages 安装模式"
		if ag.RepoRoot != "" {
			mode = "源码仓模式 · " + ag.RepoRoot
		}
		u.kv("运行模式", mode)
		u.kv("工作区根", workspaceRoot(cfg))
		u.kv("并发/队列", fmt.Sprintf("%d / %d", ag.MaxConcurrent, ag.MaxQueue))
		timeout := "不限时"
		if ag.Timeout.Duration > 0 {
			timeout = formatDur(ag.Timeout.Duration)
		}
		if ag.HeavyTimeout.Duration > 0 {
			timeout += " · heavy " + formatDur(ag.HeavyTimeout.Duration)
		}
		u.kv("任务超时", timeout)
		u.kv("时区", ag.Timezone)
	}
}

// workspaceRoot mirrors the agent runner's resolution order.
func workspaceRoot(cfg *config.Config) string {
	root := cfg.Agent.WorkspaceRoot
	if root == "" {
		root = os.Getenv("SCIFLOW_WORKSPACE_ROOT")
	}
	if root == "" {
		return "（未设置 SCIFLOW_WORKSPACE_ROOT）"
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	return root
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

func useColor() bool {
	if os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		return false
	}
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// stdinInteractive reports whether prompts can actually be answered: stdin
// must be a character device (TTY) that is not /dev/null (nohup/setsid and
// service managers attach /dev/null, which would otherwise read as EOF).
func stdinInteractive() bool {
	fi, err := os.Stdin.Stat()
	if err != nil || fi.Mode()&os.ModeCharDevice == 0 {
		return false
	}
	if devNull, err := os.Stat(os.DevNull); err == nil && os.SameFile(fi, devNull) {
		return false
	}
	return true
}

// dispWidth returns the terminal column count of s, counting East Asian
// wide runes as 2 columns (ANSI escape sequences are NOT stripped here).
func dispWidth(s string) int {
	w := 0
	for _, r := range s {
		w += runeWidth(r)
	}
	return w
}

func runeWidth(r rune) int {
	switch {
	case r >= 0x1100 && r <= 0x115F, // Hangul Jamo
		r >= 0x2E80 && r <= 0x303E, // CJK radicals / Kangxi / CJK symbols
		r >= 0x3041 && r <= 0x33FF, // Hiragana .. CJK compatibility
		r >= 0x3400 && r <= 0x4DBF, // CJK extension A
		r >= 0x4E00 && r <= 0x9FFF, // CJK unified ideographs
		r >= 0xA000 && r <= 0xA4CF, // Yi syllables
		r >= 0xAC00 && r <= 0xD7A3, // Hangul syllables
		r >= 0xF900 && r <= 0xFAFF, // CJK compatibility ideographs
		r >= 0xFE30 && r <= 0xFE4F, // CJK compatibility forms
		r >= 0xFF00 && r <= 0xFF60, // fullwidth forms
		r >= 0xFFE0 && r <= 0xFFE6,
		r >= 0x20000 && r <= 0x2FFFD,
		r >= 0x30000 && r <= 0x3FFFD:
		return 2
	}
	return 1
}

var reANSI = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripANSI(s string) string { return reANSI.ReplaceAllString(s, "") }

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out
}

func lastLine(s string) string {
	lines := nonEmptyLines(s)
	if len(lines) == 0 {
		return ""
	}
	return lines[len(lines)-1]
}

// formatDur renders durations compactly for the summary panel ("24h",
// "1h30m", "45s") instead of Go's "24h0m0s".
func formatDur(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	var b strings.Builder
	if h := int(d.Hours()); h > 0 {
		fmt.Fprintf(&b, "%dh", h)
	}
	if m := int(d.Minutes()) % 60; m > 0 {
		fmt.Fprintf(&b, "%dm", m)
	}
	if s := int(d.Seconds()) % 60; s > 0 || b.Len() == 0 {
		fmt.Fprintf(&b, "%ds", s)
	}
	return b.String()
}
