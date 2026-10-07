package agent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"log_gateway_server/internal/config"
)

// TaskStatus is the lifecycle state of a single agent invocation.
type TaskStatus string

const (
	StatusQueued  TaskStatus = "queued"
	StatusRunning TaskStatus = "running"
	StatusDone    TaskStatus = "done"
	StatusFailed  TaskStatus = "failed"
	StatusKilled  TaskStatus = "killed"
)

// Task mode selects the ScienceFlow CLI entrypoint the gateway launches:
// ModeLite runs the interactive REPL solver (cli repl), ModeHeavy runs the
// long-horizon LNR pipeline (cli run --type lnr).
const (
	ModeLite  = "lite"
	ModeHeavy = "heavy"
)

// NormalizeMode maps a user-supplied mode string onto ModeLite/ModeHeavy.
// Empty or unknown values fall back to ModeLite.
func NormalizeMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case ModeHeavy:
		return ModeHeavy
	default:
		return ModeLite
	}
}

// RawLogSource is the source name the tailer uses for per-session agent
// RAW.log files (main.go wires WatchFile/UnwatchFile under this name).
const RawLogSource = "agent-logs"

// RunDirName is the per-session execution workspace directory the gateway's
// manifest resolves to (workspace_base/run_id/exp_id with exp_id="run").
const RunDirName = "run"

// SessionInteractionLogPath returns the interaction transcript the installed
// scienceflow CLI writes for a session workspace directory
// (<sessionDir>/run/.logs/interaction.log). In --plain mode the REPL mirrors
// the full interaction (user/assistant/tool turns) into this file instead of
// stdout, so the gateway tails it alongside RAW.log.
func SessionInteractionLogPath(sessionDir string) string {
	return filepath.Join(sessionDir, RunDirName, ".logs", "interaction.log")
}

// Task is one agent invocation bound to a (user, session) pair.
type Task struct {
	ID        string
	User      string
	Session   string
	Query     string
	Mode      string
	Workspace string
	LogDir    string
	manifest  string // set by Invoke before enqueueing; read by the dispatcher
	timeout   time.Duration

	mu         sync.Mutex
	status     TaskStatus
	cmd        *exec.Cmd
	killed     bool
	started    time.Time
	finished   time.Time
	exitCode   int
	err        error
	outBuf     *ringBuffer
	rawLog     *os.File // append-only mirror of agent stdout, tailed by the gateway's own SSE
	RawLogPath string   // public path of RAW.log for API responses
}

// InteractionLogPath returns the in-workspace interaction transcript of this
// task (<workspace>/run/.logs/interaction.log), tailed by the gateway's SSE.
func (t *Task) InteractionLogPath() string {
	return filepath.Join(t.Workspace, ".logs", "interaction.log")
}

func (t *Task) Status() TaskStatus {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.status
}

// Snapshot is a point-in-time view of a Task for API responses.
type Snapshot struct {
	ID         string     `json:"id"`
	User       string     `json:"user"`
	Session    string     `json:"session"`
	Status     TaskStatus `json:"status"`
	Mode       string     `json:"mode"`
	Workspace  string     `json:"workspace"`
	LogDir     string     `json:"log_dir"`
	RawLogPath string     `json:"raw_log_path,omitempty"`
	PID        int        `json:"pid,omitempty"`
	StartedAt  string     `json:"started_at,omitempty"`
	EndedAt    string     `json:"ended_at,omitempty"`
	ExitCode   int        `json:"exit_code,omitempty"`
	Error      string     `json:"error,omitempty"`
}

func (t *Task) Snapshot() Snapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := Snapshot{
		ID:         t.ID,
		User:       t.User,
		Session:    t.Session,
		Status:     t.status,
		Mode:       t.Mode,
		Workspace:  t.Workspace,
		LogDir:     t.LogDir,
		RawLogPath: t.RawLogPath,
		ExitCode:   t.exitCode,
	}
	if t.cmd != nil && t.cmd.Process != nil {
		s.PID = t.cmd.Process.Pid
	}
	if !t.started.IsZero() {
		s.StartedAt = t.started.UTC().Format(time.RFC3339Nano)
	}
	if !t.finished.IsZero() {
		s.EndedAt = t.finished.UTC().Format(time.RFC3339Nano)
	}
	if t.err != nil {
		s.Error = t.err.Error()
	}
	return s
}

// Output returns up to maxBytes of the most recent captured stdout/stderr.
func (t *Task) Output(maxBytes int) []byte {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.outBuf == nil {
		return nil
	}
	return t.outBuf.tail(maxBytes)
}

// ErrQueueFull is returned by Invoke when the wait queue is at capacity.
var ErrQueueFull = errors.New("agent task queue is full; try again later")

// Runner owns the agent subprocess lifecycle for all sessions. It runs a
// single dispatcher goroutine that pulls tasks off a FIFO queue and starts
// them subject to the MaxConcurrent running limit. When MaxConcurrent slots
// are all occupied, new tasks wait in the queue (up to MaxQueue); exceeding
// MaxQueue causes Invoke to return ErrQueueFull immediately.
type Runner struct {
	cfg    config.AgentConfig
	logger *log.Logger
	loc    *time.Location // zone for human-facing marker timestamps (config agent.timezone)

	models *ModelStore

	mu      sync.Mutex
	running map[string]*Task // all live tasks (queued or running), keyed by task ID
	bySess  map[string]*Task // latest task per session (for status queries)

	// Lifecycle hooks, wired by main.go to drive dynamic RAW.log tailing:
	// OnTaskStart fires when a task transitions to running (subprocess about
	// to start); OnTaskFinish fires once the task reaches a terminal state.
	OnTaskStart  func(t *Task)
	OnTaskFinish func(t *Task)

	queue    chan *Task    // buffered FIFO; capacity = MaxQueue
	dispatch chan struct{} // capacity-1 signal to wake the dispatcher
	stop     chan struct{}
	wg       sync.WaitGroup
}

func New(cfg config.AgentConfig, logger *log.Logger) *Runner {
	maxQueue := cfg.MaxQueue
	if maxQueue <= 0 {
		maxQueue = 16
	}
	models, err := NewModelStore(cfg.ModelStorePath)
	if err != nil && logger != nil {
		logger.Printf("load model store: %v", err)
	}
	// Stop-marker timestamps must match the user's wall clock. The container
	// TZ is unreliable (resets to UTC across restarts), so the zone comes from
	// config explicitly.
	loc, locErr := time.LoadLocation(cfg.Timezone)
	if locErr != nil || loc == nil {
		loc = time.UTC
		if logger != nil {
			logger.Printf("invalid agent.timezone %q: %v; falling back to UTC", cfg.Timezone, locErr)
		}
	}
	r := &Runner{
		cfg:      cfg,
		logger:   logger,
		loc:      loc,
		models:   models,
		running:  make(map[string]*Task),
		bySess:   make(map[string]*Task),
		queue:    make(chan *Task, maxQueue),
		dispatch: make(chan struct{}, 1),
		stop:     make(chan struct{}),
	}
	if cfg.Enabled {
		r.wg.Add(1)
		go r.dispatchLoop()
	}
	return r
}

// WorkspaceFor resolves the execution workspace for a (user, session) pair:
// $SCIFLOW_WORKSPACE_ROOT/<user>/<session> (or agent.workspace_root override).
func (r *Runner) WorkspaceFor(user, session string) (string, error) {
	root, err := r.rootPath()
	if err != nil {
		return "", err
	}
	ws := filepath.Join(root, user, session)
	if err := os.MkdirAll(ws, 0o755); err != nil {
		return "", fmt.Errorf("create workspace: %w", err)
	}
	return ws, nil
}

// rootPath returns the workspace root from config or $SCIFLOW_WORKSPACE_ROOT.
func (r *Runner) rootPath() (string, error) {
	root := r.cfg.WorkspaceRoot
	if root == "" {
		root = os.Getenv("SCIFLOW_WORKSPACE_ROOT")
	}
	if root == "" {
		return "", errors.New("SCIFLOW_WORKSPACE_ROOT is not set and agent.workspace_root is empty")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return "", fmt.Errorf("resolve workspace root: %w", err)
	}
	return root, nil
}

// RootPath returns the workspace root resolved from config or
// $SCIFLOW_WORKSPACE_ROOT. Used by the server to locate a session's workspace
// directory for file-tree sync.
func (r *Runner) RootPath() (string, error) {
	return r.rootPath()
}

// ModelStore returns the runtime model configuration store.
func (r *Runner) ModelStore() *ModelStore {
	return r.models
}

// Current returns the latest task for a session, if any.
func (r *Runner) Current(session string) *Task {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.bySess[session]
}

// Get retrieves a task by ID.
func (r *Runner) Get(id string) *Task {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running[id]
}

// QueueStats reports the current number of queued and running tasks.
func (r *Runner) QueueStats() (queued, running int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, t := range r.running {
		switch t.Status() {
		case StatusQueued:
			queued++
		case StatusRunning:
			running++
		}
	}
	return
}

// Invoke launches a ScienceFlow agent for the given user message. mode selects
// the task type: ModeLite runs the interactive REPL solver from a per-task
// manifest (cli repl), ModeHeavy runs the long-horizon LNR pipeline
// (cli run --type lnr). It enqueues the task and returns immediately; the
// dispatcher goroutine starts the subprocess when a MaxConcurrent slot is
// free. Returns ErrQueueFull if the wait queue is full.
func (r *Runner) Invoke(ctx context.Context, user, session, query, mode string) (*Task, error) {
	if !r.cfg.Enabled {
		return nil, errors.New("agent invocation is disabled")
	}
	if strings.TrimSpace(query) == "" {
		return nil, errors.New("query is empty")
	}
	mode = NormalizeMode(mode)

	ws, err := r.WorkspaceFor(user, session)
	if err != nil {
		return nil, err
	}
	execDir := filepath.Join(ws, "run")
	logDir := filepath.Join(execDir, r.cfg.LogDirName)

	// RAW.log lives at <workspace_root>/task_logs/<user>/<session>/RAW.log so
	// the gateway's own tailer (glob task_logs/*/*/RAW.log) can discover and
	// stream it to SSE consumers in real time.
	root, _ := r.rootPath()
	rawLogDir := filepath.Join(root, "task_logs", user, session)
	if err := os.MkdirAll(rawLogDir, 0o755); err != nil {
		return nil, fmt.Errorf("create raw log dir: %w", err)
	}
	rawLogPath := filepath.Join(rawLogDir, "RAW.log")

	// Lite tasks drive `cli repl` through a per-invocation manifest. Heavy
	// tasks pass the query directly to `cli run --type lnr` (no manifest).
	manifestPath := ""
	if mode == ModeLite {
		manifestPath, err = r.writeManifest(ws, user, session, query)
		if err != nil {
			return nil, err
		}
	}

	id := newTaskID()
	t := &Task{
		ID:         id,
		User:       user,
		Session:    session,
		Query:      query,
		Mode:       mode,
		Workspace:  execDir,
		LogDir:     logDir,
		RawLogPath: rawLogPath,
		manifest:   manifestPath,
		timeout:    r.timeoutFor(mode),
		outBuf:     newRingBuffer(512 * 1024),
	}
	t.status = StatusQueued

	r.mu.Lock()
	r.running[id] = t
	r.bySess[session] = t
	r.mu.Unlock()

	// Enqueue for the dispatcher. The channel has capacity MaxQueue so this
	// should not block, but use a non-blocking send as a safety net.
	select {
	case r.queue <- t:
	default:
		r.mu.Lock()
		delete(r.running, id)
		r.mu.Unlock()
		return nil, ErrQueueFull
	}

	// Wake the dispatcher.
	select {
	case r.dispatch <- struct{}{}:
	default:
	}

	// Watchdog: enforce per-task timeout spanning queue wait + execution.
	if t.timeout > 0 {
		go r.watchTimeout(t)
	}

	return t, nil
}

// Stop kills the task's process if still running. If the task is still
// queued, it is marked as killed so the dispatcher will skip it.
func (r *Runner) Stop(id string) bool {
	r.mu.Lock()
	t, ok := r.running[id]
	r.mu.Unlock()
	if !ok {
		log.Printf("[agent-debug] stop id=%s found=false", id)
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	pid := 0
	if t.cmd != nil && t.cmd.Process != nil {
		pid = t.cmd.Process.Pid
	}
	log.Printf("[agent-debug] stop id=%s status=%s killed=%v has_cmd=%v pid=%d", id, t.status, t.killed, t.cmd != nil, pid)
	t.killed = true
	if t.cmd != nil && t.cmd.Process != nil {
		err := killProcessTree(t.cmd)
		log.Printf("[agent-debug] kill-process-tree id=%s pid=%d err=%v", id, t.cmd.Process.Pid, err)
		if err == nil {
			t.status = StatusKilled
			t.finished = time.Now()
		}
		return err == nil
	}
	if t.status == StatusQueued {
		t.status = StatusKilled
		t.finished = time.Now()
	}
	// StatusRunning with no live process yet (cmd.Start() not reached): the
	// killed flag makes runTask terminate the process right after it starts.
	return true
}

func killProcessTree(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return nil
	}
	if runtime.GOOS == "windows" {
		out, err := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid)).CombinedOutput()
		if err != nil {
			return fmt.Errorf("taskkill pid %d: %w (%s)", cmd.Process.Pid, err, strings.TrimSpace(string(out)))
		}
		return nil
	}
	return cmd.Process.Kill()
}

// workDir resolves the subprocess working directory. When agent.repo_root is
// configured, it is used as-is so the in-repo scienceflow package shadows
// site-packages (repo mode). When empty (installed-package mode), the task's
// run workspace is used instead — crucially NOT the ScienceFlow source repo —
// so `python -m scienceflow.cli` resolves the pip-installed module. The
// workspace is created up front because exec.Cmd requires an existing cwd.
func (r *Runner) workDir(t *Task) string {
	if root := strings.TrimSpace(r.cfg.RepoRoot); root != "" {
		return root
	}
	_ = os.MkdirAll(t.Workspace, 0o755)
	return t.Workspace
}

// timeoutFor returns the wall-clock cap for a task of the given mode: heavy
// tasks use agent.heavy_timeout when set, everything else agent.timeout.
func (r *Runner) timeoutFor(mode string) time.Duration {
	if mode == ModeHeavy && r.cfg.HeavyTimeout.Duration > 0 {
		return r.cfg.HeavyTimeout.Duration
	}
	return r.cfg.Timeout.Duration
}

// watchTimeout enforces the per-invocation wall-clock timeout spanning both
// queue wait time and actual execution.
func (r *Runner) watchTimeout(t *Task) {
	timer := time.NewTimer(t.timeout)
	defer timer.Stop()
	select {
	case <-timer.C:
		t.mu.Lock()
		if t.status == StatusDone || t.status == StatusFailed || t.status == StatusKilled {
			t.mu.Unlock()
			return
		}
		t.err = fmt.Errorf("agent timed out after %s (including queue wait)", t.timeout)
		t.killed = true
		if t.cmd != nil && t.cmd.Process != nil {
			_ = t.cmd.Process.Kill()
		}
		if t.status == StatusQueued {
			t.status = StatusFailed
			t.finished = time.Now()
		}
		t.mu.Unlock()
		r.finalize(t)
	case <-r.stop:
		return
	}
}

// dispatchLoop is the single goroutine that owns the queue -> execution
// pipeline. It maintains at most MaxConcurrent concurrent subprocesses.
func (r *Runner) dispatchLoop() {
	defer r.wg.Done()
	for {
		select {
		case <-r.stop:
			return
		case <-r.dispatch:
			r.tryStartPending()
		}
	}
}

// tryStartPending drains the queue as long as there are free concurrency
// slots and pending tasks.
func (r *Runner) tryStartPending() {
	for {
		r.mu.Lock()
		runningCount := 0
		for _, t := range r.running {
			if t.Status() == StatusRunning {
				runningCount++
			}
		}
		limit := r.cfg.MaxConcurrent
		if limit <= 0 {
			limit = 1
		}
		if runningCount >= limit {
			r.mu.Unlock()
			return
		}
		r.mu.Unlock()

		select {
		case t := <-r.queue:
			r.startOne(t)
		default:
			return
		}
	}
}

// startOne launches the subprocess for a dequeued task, unless it was already
// killed/failed while waiting in the queue.
func (r *Runner) startOne(t *Task) {
	t.mu.Lock()
	if t.status != StatusQueued {
		t.mu.Unlock()
		r.finalize(t)
		return
	}
	t.status = StatusRunning
	t.started = time.Now()
	t.mu.Unlock()

	if r.OnTaskStart != nil {
		r.OnTaskStart(t)
	}

	go r.runTask(t)

	// Kick the dispatcher again in case more slots are free.
	select {
	case r.dispatch <- struct{}{}:
	default:
	}
}

// runTask executes the agent subprocess. Called in its own goroutine.
func (r *Runner) runTask(t *Task) {
	// agent.python may be a compound command like "uv run python" or
	// "python3 -u". exec.Command treats the whole string as a single
	// executable name, so split it into argv first.
	argv := splitCommand(r.cfg.Python)
	if len(argv) == 0 {
		t.mu.Lock()
		t.err = fmt.Errorf("agent.python is empty")
		t.status = StatusFailed
		t.finished = time.Now()
		t.mu.Unlock()
		r.finalize(t)
		return
	}
	// For uv-based commands, skip environment sync: dependency resolution can
	// hit the network and stall the gateway. The venv must be pre-synced
	// (uv sync) as part of deployment.
	if argv[0] == "uv" && containsToken(argv, "run") && !containsToken(argv, "--no-sync") && !containsToken(argv, "--frozen") {
		argv = insertAfter(argv, "run", "--no-sync")
	}

	args := append(argv[1:], r.buildArgs(t)...)
	cmd := exec.Command(argv[0], args...)
	cmd.Dir = r.workDir(t)
	cmd.Env = r.buildEnv(t)

	// Open RAW.log in append mode so multiple invocations of the same session
	// accumulate a single transcript. The tailer picks up new bytes in real time.
	rawLog, rawErr := os.OpenFile(t.RawLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if rawErr != nil {
		// Non-fatal: the ring buffer still captures output for the status API.
		r.logger.Printf("open raw log %s: %v", t.RawLogPath, rawErr)
	} else {
		// Write a header so the transcript is self-describing, including the
		// exact command about to be executed (helps diagnosing startup hangs).
		fmt.Fprintf(rawLog, "\n=== [%s] task=%s mode=%s query=%q ===\n=== cmd: %s %s ===\n",
			time.Now().UTC().Format(time.RFC3339Nano), t.ID, t.Mode, t.Query,
			argv[0], strings.Join(append(append([]string(nil), argv[1:]...), r.buildArgs(t)...), " "))
	}
	t.mu.Lock()
	t.rawLog = rawLog
	t.mu.Unlock()
	defer func() {
		if rawLog != nil {
			_ = rawLog.Close()
		}
	}()

	pr, pw, err := os.Pipe()
	if err != nil {
		t.mu.Lock()
		t.err = err
		t.status = StatusFailed
		t.finished = time.Now()
		t.mu.Unlock()
		r.finalize(t)
		return
	}
	cmd.Stdout = pw
	cmd.Stderr = pw
	t.mu.Lock()
	t.cmd = cmd
	t.mu.Unlock()

	doneCopy := make(chan struct{})
	go func() {
		buf := make([]byte, 4096)
		for {
			n, err := pr.Read(buf)
			if n > 0 {
				t.mu.Lock()
				t.outBuf.write(buf[:n])
				if t.rawLog != nil {
					_, _ = t.rawLog.Write(buf[:n])
				}
				t.mu.Unlock()
			}
			if err != nil {
				break
			}
		}
		_ = pr.Close()
		close(doneCopy)
	}()

	if err := cmd.Start(); err != nil {
		_ = pw.Close()
		_ = pr.Close()
		<-doneCopy
		// Surface the failure in RAW.log too, so a task that never produced
		// output doesn't look like a silent hang.
		if rawLog != nil {
			fmt.Fprintf(rawLog, "=== FAILED TO START: %v ===\n", err)
		}
		t.mu.Lock()
		t.err = err
		t.status = StatusFailed
		t.finished = time.Now()
		t.mu.Unlock()
		r.finalize(t)
		return
	}
	_ = pw.Close()

	// A Stop request may have arrived between status=Running and the actual
	// process start; honor it immediately instead of running the full task.
	t.mu.Lock()
	killed := t.killed
	t.mu.Unlock()
	if killed {
		_ = killProcessTree(cmd)
	}

	err = cmd.Wait()
	<-doneCopy

	t.mu.Lock()
	t.finished = time.Now()
	t.exitCode = cmd.ProcessState.ExitCode()
	switch {
	case t.killed:
		t.status = StatusKilled
	case err != nil || t.exitCode != 0:
		t.status = StatusFailed
		if err != nil {
			t.err = err
		} else {
			t.err = fmt.Errorf("agent exited with code %d", t.exitCode)
		}
	default:
		t.status = StatusDone
	}
	t.mu.Unlock()
	// Mark manual termination in RAW.log so the chat history (parsed from the
	// transcript on backfill) shows the run was stopped by the user instead of
	// ending naturally. Written after doneCopy so it never interleaves with
	// subprocess output. Uses the configured display timezone (agent.timezone,
	// default Asia/Shanghai) — NOT container-local time, which resets to UTC
	// across container restarts.
	if t.killed && rawLog != nil {
		_, _ = fmt.Fprintf(rawLog, "\n> 手动终止输出 · %s\n", time.Now().In(r.loc).Format("2006-01-02 15:04:05"))
	}
	r.finalize(t)
}

// finalize removes a terminal task from the running map and kicks the
// dispatcher so a queued task can take the freed slot.
func (r *Runner) finalize(t *Task) {
	r.mu.Lock()
	delete(r.running, t.ID)
	r.mu.Unlock()
	if r.OnTaskFinish != nil {
		r.OnTaskFinish(t)
	}
	select {
	case r.dispatch <- struct{}{}:
	default:
	}
}

// writeManifest emits a single-task YAML manifest consumed by `cli repl -m`.
func (r *Runner) writeManifest(ws, user, session, query string) (string, error) {
	absWS, err := filepath.Abs(ws)
	if err != nil {
		return "", fmt.Errorf("resolve workspace path: %w", err)
	}
	queryFile := filepath.Join(absWS, "gateway_first_user.txt")
	if err := os.WriteFile(queryFile, []byte(query), 0o644); err != nil {
		return "", fmt.Errorf("write first-user query: %w", err)
	}
	manifest := filepath.Join(absWS, "gateway_manifest.yaml")

	var b strings.Builder
	b.WriteString("defaults:\n")
	fmt.Fprintf(&b, "  repl_auto_first_user: true\n  repl_exit_after_auto: true\n  repl_profile: lite\n")
	fmt.Fprintf(&b, "  repl_first_user_query_file: %s\n", queryFile)
	fmt.Fprintf(&b, "  workspace_base: %s\n", filepath.Dir(ws))
	b.WriteString("tasks:\n")
	fmt.Fprintf(&b, "  - exp_id: run\n")
	fmt.Fprintf(&b, "    run_id: %s\n", session)
	fmt.Fprintf(&b, "    task: %q\n", query)
	if r.cfg.InputDataDir != "" {
		fmt.Fprintf(&b, "    input_data_dir: %s\n", r.cfg.InputDataDir)
	}

	if err := os.WriteFile(manifest, []byte(b.String()), 0o644); err != nil {
		return "", fmt.Errorf("write manifest: %w", err)
	}
	return manifest, nil
}

// splitCommand splits a command string like `uv run python` into argv tokens,
// honoring double quotes for paths containing spaces.
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
		c := s[i]
		switch {
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

// containsToken reports whether token is present in argv.
func containsToken(argv []string, token string) bool {
	for _, a := range argv {
		if a == token {
			return true
		}
	}
	return false
}

// insertAfter inserts token right after the first occurrence of anchor.
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

// buildArgs builds the scienceflow CLI arguments for a task. Lite tasks use
// the configured command (repl by default) with the per-task manifest; heavy
// tasks always run the long-horizon pipeline `cli run --type lnr` with the
// query passed via --task.
func (r *Runner) buildArgs(t *Task) []string {
	mod := r.cfg.Module
	if mod == "" {
		mod = "scienceflow.cli"
	}
	if t.Mode == ModeHeavy {
		args := []string{"-m", mod, "run", "--type", "lnr",
			"-t", t.Query,
			"-w", t.Workspace,
		}
		if r.cfg.ConfigYAML != "" {
			args = append(args, "-c", r.cfg.ConfigYAML)
		}
		if r.cfg.InputDataDir != "" {
			args = append(args, "-d", r.cfg.InputDataDir)
		}
		return args
	}
	cmd := r.cfg.Command
	if cmd == "" {
		cmd = "repl"
	}
	args := []string{"-m", mod, cmd, "-m", t.manifest}
	if r.cfg.ConfigYAML != "" {
		args = append(args, "-c", r.cfg.ConfigYAML)
	}
	if cmd == "repl" {
		args = append(args, "--auto-first-user", "--exit-after-auto", "--plain")
	}
	if r.cfg.ExpID != "" {
		args = append(args, "--exp-id", r.cfg.ExpID)
	}
	return args
}

func (r *Runner) buildEnv(t *Task) []string {
	env := os.Environ()
	env = append(env,
		"SCIFLOW_USER="+t.User,
		"SCIFLOW_SESSION="+t.Session,
		"SCIFLOW_TASK_ID="+t.ID,
		"SCIFLOW_TASK_MODE="+t.Mode,
		"SCIFLOW_TASK_WORKSPACE="+t.Workspace,
	)
	if r.models != nil {
		// Stage models: the installed CLI ignores CODE_*/FEEDBACK_* env vars
		// and resolves endpoints from the model registry instead. When a
		// session overrides the code/feedback stages, materialize a per-task
		// registry copy with those defaults and point the subprocess at it
		// via XDG_CONFIG_HOME ($XDG_CONFIG_HOME/scienceflow/models.json).
		if code, feedback, ok := r.models.SessionStageAliases(t.Session); ok {
			if dir, err := r.sessionXDGConfigDir(t); err == nil {
				if err := r.models.WriteSessionRegistry(dir, code, feedback); err != nil {
					if r.logger != nil {
						r.logger.Printf("write session model registry: %v", err)
					}
				} else {
					env = append(env, "XDG_CONFIG_HOME="+dir)
				}
			} else if r.logger != nil {
				r.logger.Printf("resolve session model registry dir: %v", err)
			}
		}
	}
	root := r.cfg.WorkspaceRoot
	if root == "" {
		root = os.Getenv("SCIFLOW_WORKSPACE_ROOT")
	}
	if root != "" {
		root, err := filepath.Abs(root)
		if err == nil {
			env = append(env, "SCIFLOW_WORKSPACE_ROOT="+root)
		}
	}
	env = append(env, "PYTHONUNBUFFERED=1")
	return env
}

// sessionXDGConfigDir returns a private XDG_CONFIG_HOME for one agent task.
// It lives under <workspace_root>/task_logs/<user>/<session>/xdg — outside
// both the CLI's git-tracked run workspace and the file-tree sync root, so
// the registry copy (which contains API keys) is never committed or listed.
func (r *Runner) sessionXDGConfigDir(t *Task) (string, error) {
	root, err := r.rootPath()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, "task_logs", t.User, t.Session, "xdg")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

func newTaskID() string {
	return fmt.Sprintf("task-%d", time.Now().UnixNano())
}
