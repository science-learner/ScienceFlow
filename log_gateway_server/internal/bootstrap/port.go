package bootstrap

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"log_gateway_server/internal/config"
)

// defaultStartPort is the default offered by the interactive startup-port
// prompt (the user can always type another port; non-interactive runs use
// config server.port instead).
const defaultStartPort = 8901

// PortOwner identifies the process listening on a TCP port.
type PortOwner struct {
	PID  int
	Name string // process/image name (short form)
	Cmd  string // full command line when available (Linux: /proc/<pid>/cmdline)
}

// ---------------------------------------------------------------------------
// Port probing
// ---------------------------------------------------------------------------

// listenAddr formats host:port for the bind probe; empty host binds all
// interfaces.
func listenAddr(host string, port int) string {
	if strings.TrimSpace(host) == "" {
		host = "0.0.0.0"
	}
	return net.JoinHostPort(host, strconv.Itoa(port))
}

// portAvailable reports whether the gateway can bind host:port right now.
// It performs a real listen (then closes it), which is the only reliable
// check that also covers sockets owned by other users.
func portAvailable(host string, port int) bool {
	ln, err := net.Listen("tcp", listenAddr(host, port))
	if err != nil {
		return false
	}
	_ = ln.Close()
	return true
}

// waitPortFree polls portAvailability until the port is free or the timeout
// elapses.
func waitPortFree(host string, port int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if portAvailable(host, port) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// ---------------------------------------------------------------------------
// Interactive port selection & conflict resolution
// ---------------------------------------------------------------------------

// askPort asks for a TCP port with def pre-filled; invalid input re-prompts
// (at most 3 times, then def).
func askPort(u *ui, stdin *bufio.Reader, def int) int {
	for i := 0; i < 3; i++ {
		raw := u.ask("启动端口", strconv.Itoa(def), stdin)
		p, err := strconv.Atoi(strings.TrimSpace(raw))
		if err == nil && p >= 1 && p <= 65535 {
			return p
		}
		u.warn("无效端口 %q（需为 1-65535 的整数）", raw)
	}
	return def
}

// resolvePort ensures cfg.Server.Port is bindable. On a conflict it shows the
// occupying process and — in interactive runs only — offers to terminate it
// after explicit confirmation (default No). Non-interactive runs abort with
// the manual command instead of touching any process.
func resolvePort(u *ui, stdin *bufio.Reader, interactive bool, cfg *config.Config) error {
	host := cfg.Server.Host
	for {
		port := cfg.Server.Port
		if portAvailable(host, port) {
			u.ok("端口 %d 可用", port)
			return nil
		}
		owner, lookupErr := lookupPortOwner(port)
		if lookupErr != nil || owner == nil {
			u.warn("端口 %d 已被占用，但无法识别占用进程", port)
			if lookupErr != nil {
				u.quiet("%v", lookupErr)
			}
			if !interactive {
				u.blank()
				u.fail("请手动释放端口 %d 后重新启动", port)
				return fmt.Errorf("%w: port %d is in use", ErrAborted, port)
			}
			cfg.Server.Port = askPort(u, stdin, port)
			continue
		}

		u.warn("端口 %d 已被以下进程占用", port)
		u.kv("进程", fmt.Sprintf("%s（PID %d）", owner.Name, owner.PID))
		if owner.Cmd != "" {
			u.kv("命令", clip(owner.Cmd, 100))
		}
		if !interactive {
			u.blank()
			u.fail("非交互模式不自动中止占用进程；请手动执行后重新启动：")
			u.print("      %s", u.paint(cBold, killHint(owner.PID)))
			return fmt.Errorf("%w: port %d occupied by pid %d", ErrAborted, port, owner.PID)
		}
		if !u.confirm(fmt.Sprintf("是否中止该进程以释放端口 %d？", port), false, stdin) {
			u.blank()
			u.fail("已取消中止进程。请手动释放端口 %d 后重新启动：", port)
			u.print("      %s", u.paint(cBold, killHint(owner.PID)))
			return fmt.Errorf("%w: kill declined", ErrAborted)
		}
		if err := terminateOwner(u, host, port, owner); err != nil {
			return fmt.Errorf("%w: %v", ErrAborted, err)
		}
	}
}

// terminateOwner stops the confirmed port owner and waits for the port to be
// released. Safety measures: numeric PIDs only (never shell-evaluated), refuse
// pid < 2 and our own pid, re-verify that the same pid still listens on the
// port right before signalling, escalate SIGTERM -> SIGKILL on timeout.
func terminateOwner(u *ui, host string, port int, owner *PortOwner) error {
	if owner.PID < 2 || owner.PID == os.Getpid() {
		u.fail("拒绝中止 PID %d（不安全的进程号）", owner.PID)
		return fmt.Errorf("unsafe pid %d", owner.PID)
	}

	// PID reuse / stale info guard: re-resolve the owner immediately before
	// sending any signal.
	cur, err := lookupPortOwner(port)
	if err == nil {
		if cur == nil {
			if portAvailable(host, port) {
				return nil // freed meanwhile
			}
			u.fail("端口占用者信息已失效，请重试")
			return fmt.Errorf("port owner no longer identifiable")
		}
		if cur.PID != owner.PID {
			u.fail("端口占用进程已变化（PID %d → %d），已取消操作", owner.PID, cur.PID)
			return fmt.Errorf("port owner changed to pid %d", cur.PID)
		}
	}

	if err := terminateProcess(owner.PID); err != nil {
		u.fail("中止进程 %d 失败：%v", owner.PID, err)
		u.quiet("可能权限不足（进程属于其他用户），请手动执行：%s", killHint(owner.PID))
		return fmt.Errorf("terminate pid %d: %w", owner.PID, err)
	}
	u.quiet("已发送终止信号，等待端口 %d 释放…", port)
	if waitPortFree(host, port, 3*time.Second) {
		u.ok("端口 %d 已释放", port)
		return nil
	}

	u.quiet("进程未退出，发送强制终止信号…")
	if err := forceKillProcess(owner.PID); err != nil {
		u.fail("强制中止进程 %d 失败：%v", owner.PID, err)
		return fmt.Errorf("force kill pid %d: %w", owner.PID, err)
	}
	if waitPortFree(host, port, 3*time.Second) {
		u.ok("端口 %d 已释放", port)
		return nil
	}
	u.fail("端口 %d 仍被占用", port)
	return fmt.Errorf("port %d still occupied after kill", port)
}

// ---------------------------------------------------------------------------
// Occupier parsing (pure helpers, unit-tested)
// ---------------------------------------------------------------------------

var (
	reSSOwnerNamed = regexp.MustCompile(`\("([^"]*)",pid=(\d+)`)
	reSSOwnerPID   = regexp.MustCompile(`pid=(\d+)`)
)

// parseSSOwners extracts unique listening-socket owners for port from
// `ss -ltnp` output, e.g.:
//
//	LISTEN 0 4096 0.0.0.0:8901 0.0.0.0:* users:(("python3",pid=4567,fd=3))
func parseSSOwners(out string, port int) []PortOwner {
	suffix := fmt.Sprintf(":%d", port)
	seen := map[int]bool{}
	var owners []PortOwner
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, suffix) {
			continue
		}
		pid, name := 0, ""
		if m := reSSOwnerNamed.FindStringSubmatch(line); m != nil {
			pid, _ = strconv.Atoi(m[2])
			name = m[1]
		} else if m := reSSOwnerPID.FindStringSubmatch(line); m != nil {
			pid, _ = strconv.Atoi(m[1])
		}
		if pid > 0 && !seen[pid] {
			seen[pid] = true
			if name == "" {
				name = "?"
			}
			owners = append(owners, PortOwner{PID: pid, Name: name})
		}
	}
	return owners
}

// parseLsofOwners extracts unique owners from `lsof -nP -iTCP:<port>
// -sTCP:LISTEN` output (first two columns are COMMAND and PID).
func parseLsofOwners(out string) []PortOwner {
	seen := map[int]bool{}
	var owners []PortOwner
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || strings.EqualFold(fields[0], "COMMAND") {
			continue
		}
		pid, err := strconv.Atoi(fields[1])
		if err != nil || pid <= 0 || seen[pid] {
			continue
		}
		seen[pid] = true
		owners = append(owners, PortOwner{PID: pid, Name: fields[0]})
	}
	return owners
}

// parseNetstatOwners extracts unique LISTENING pids for port from
// `netstat -ano -p tcp` output (Windows).
func parseNetstatOwners(out string, port int) []PortOwner {
	suffix := fmt.Sprintf(":%d", port)
	seen := map[int]bool{}
	var owners []PortOwner
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		listening := false
		for _, f := range fields {
			if strings.EqualFold(f, "LISTENING") {
				listening = true
				break
			}
		}
		if !listening || !strings.HasSuffix(fields[1], suffix) {
			continue
		}
		pid, err := strconv.Atoi(fields[len(fields)-1])
		if err != nil || pid <= 0 || seen[pid] {
			continue
		}
		seen[pid] = true
		owners = append(owners, PortOwner{PID: pid})
	}
	return owners
}

// parseTasklistImage extracts the image name from `tasklist /FO CSV /NH`
// output (first CSV field of the first data line).
func parseTasklistImage(out string) string {
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if i := strings.Index(line, "\","); i > 0 {
			return strings.Trim(line[:i], `"`)
		}
		return strings.Trim(line, `"`)
	}
	return ""
}

// lsofArgs returns the lsof arguments that list listeners on port.
func lsofArgs(port int) []string {
	return []string{"-nP", fmt.Sprintf("-iTCP:%d", port), "-sTCP:LISTEN"}
}

// parseFuserOwners extracts pids from `fuser -n tcp <port>` stdout, which is
// a plain space-separated pid list (the "8901/tcp:" label goes to stderr).
func parseFuserOwners(out string) []PortOwner {
	seen := map[int]bool{}
	var owners []PortOwner
	for _, f := range strings.Fields(out) {
		pid, err := strconv.Atoi(f)
		if err != nil || pid <= 0 || seen[pid] {
			continue
		}
		seen[pid] = true
		owners = append(owners, PortOwner{PID: pid})
	}
	return owners
}

// finalizeOwner enriches an owner with the full command line (Linux /proc)
// and, when the short name is missing, derives the name from the command.
func finalizeOwner(o *PortOwner) *PortOwner {
	if o.Cmd == "" {
		o.Cmd = processCommand(o.PID)
	}
	if (o.Name == "" || o.Name == "?") && o.Cmd != "" {
		if fields := strings.Fields(o.Cmd); len(fields) > 0 {
			o.Name = filepath.Base(fields[0])
		}
	}
	if o.Name == "" {
		o.Name = "?"
	}
	return o
}

// clip truncates s to max runes for display, appending an ellipsis.
func clip(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max-1]) + "…"
}
