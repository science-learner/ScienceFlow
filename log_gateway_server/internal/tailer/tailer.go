package tailer

import (
	"bytes"
	"encoding/base64"
	"errors"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"log_gateway_server/internal/config"
	"log_gateway_server/internal/hub"
)

// maxLineBytes is a safety cap for a single line; if a file has no newline for
// longer than this, the buffered bytes are emitted as one message anyway.
const maxLineBytes = 4 * 1024 * 1024

// rawChunkSize is how many bytes are emitted per event in raw mode.
const rawChunkSize = 32 * 1024

// Tailer discovers log files via glob patterns and keeps one harvester running
// per file, mirroring Filebeat's prospector/harvester model.
type Tailer struct {
	inputs   []config.InputConfig
	registry *Registry
	hub      *hub.Hub
	logger   *log.Logger

	mu          sync.Mutex
	harvesters  map[string]*harvester
	watched     map[string]config.InputConfig // explicit per-file watches, keyed by abs path
	dynNames    map[string]bool               // dynamic source names without a static glob backing
	pendingUn   map[string]*time.Timer        // scheduled unwatch timers, keyed by abs path
	kick        chan struct{}                 // wakes scanLoop immediately

	stop chan struct{}
	wg   sync.WaitGroup
}

func New(inputs []config.InputConfig, registry *Registry, h *hub.Hub, logger *log.Logger) *Tailer {
	return &Tailer{
		inputs:     inputs,
		registry:   registry,
		hub:        h,
		logger:     logger,
		harvesters: make(map[string]*harvester),
		watched:    make(map[string]config.InputConfig),
		dynNames:   make(map[string]bool),
		pendingUn:  make(map[string]*time.Timer),
		kick:       make(chan struct{}, 1),
		stop:       make(chan struct{}),
	}
}

func (t *Tailer) Start(scanFrequency time.Duration) {
	t.wg.Add(1)
	go t.scanLoop(scanFrequency)
}

// AddInput appends an input config at runtime (e.g. an agent workspace log
// directory discovered after start). Safe to call concurrently with scan.
func (t *Tailer) AddInput(in config.InputConfig) {
	t.mu.Lock()
	t.inputs = append(t.inputs, in)
	t.mu.Unlock()
}

// RegisterSourceName makes a source name visible to /sources and ACL filtering
// without registering any file glob for it. Used for dynamic sources whose
// files are attached later via WatchFile (e.g. "agent-logs").
func (t *Tailer) RegisterSourceName(name string) {
	if name == "" {
		return
	}
	t.mu.Lock()
	t.dynNames[name] = true
	t.mu.Unlock()
}

// WatchFile registers one concrete file for tailing under the given source
// name, kicking an immediate scan so the harvester attaches without waiting
// for the next scan tick. Idempotent; cancels any pending unwatch for the
// same path. This is the dynamic counterpart of static glob inputs: the
// gateway tracks exactly the RAW.log of the (user, session) that invoked,
// rather than pre-configured directories.
func (t *Tailer) WatchFile(path, name string, poll time.Duration) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return
	}
	abs = filepath.ToSlash(abs)
	// Capture the file size NOW — at registration time. The harvester attaches
	// asynchronously (next scan tick), so any bytes written in between must be
	// tailed too; without this baseline they would be skipped silently.
	var startOffset int64
	if fi, err := os.Stat(abs); err == nil && !fi.IsDir() {
		startOffset = fi.Size()
	}
	in := config.InputConfig{
		Name:         name,
		Path:         abs,
		Mode:         "line",
		TailFiles:    true,
		PollInterval: config.Duration{Duration: poll},
		// A brand-new file (or a task that wants its full transcript) starts
		// from the registration baseline; existing content stays un-tailed.
		StartOffset:    startOffset,
		HasStartOffset: true,
	}
	t.mu.Lock()
	if tmr, ok := t.pendingUn[abs]; ok {
		tmr.Stop()
		delete(t.pendingUn, abs)
	}
	t.watched[abs] = in
	if name != "" {
		t.dynNames[name] = true
	}
	t.mu.Unlock()
	t.kickScan()
}

// UnwatchFile removes a file watch immediately (harvester stops on next scan).
func (t *Tailer) UnwatchFile(path string) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return
	}
	abs = filepath.ToSlash(abs)
	t.mu.Lock()
	if tmr, ok := t.pendingUn[abs]; ok {
		tmr.Stop()
		delete(t.pendingUn, abs)
	}
	if _, ok := t.watched[abs]; ok {
		delete(t.watched, abs)
	}
	t.mu.Unlock()
	t.kickScan()
}

// UnwatchFileAfter schedules UnwatchFile after delay. Used to keep tailing a
// finished task's RAW.log briefly so the final lines flush to SSE before the
// harvester is detached. Re-watching the same path cancels the pending timer.
func (t *Tailer) UnwatchFileAfter(path string, delay time.Duration) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return
	}
	abs = filepath.ToSlash(abs)
	t.mu.Lock()
	if _, ok := t.watched[abs]; !ok {
		t.mu.Unlock()
		return
	}
	if old, ok := t.pendingUn[abs]; ok {
		old.Stop()
	}
	tmr := time.AfterFunc(delay, func() {
		t.mu.Lock()
		delete(t.watched, abs)
		delete(t.pendingUn, abs)
		t.mu.Unlock()
		t.kickScan()
	})
	t.pendingUn[abs] = tmr
	t.mu.Unlock()
}

func (t *Tailer) kickScan() {
	select {
	case t.kick <- struct{}{}:
	default:
	}
}

func (t *Tailer) scanLoop(freq time.Duration) {
	defer t.wg.Done()
	t.scan()
	ticker := time.NewTicker(freq)
	defer ticker.Stop()
	for {
		select {
		case <-t.stop:
			return
		case <-ticker.C:
			t.scan()
		case <-t.kick:
			t.scan()
		}
	}
}

func (t *Tailer) scan() {
	found := make(map[string]config.InputConfig)
	t.mu.Lock()
	inputs := append([]config.InputConfig(nil), t.inputs...)
	watched := make(map[string]config.InputConfig, len(t.watched))
	for p, in := range t.watched {
		watched[p] = in
	}
	t.mu.Unlock()
	for _, in := range inputs {
		paths, err := filepath.Glob(in.Path)
		if err != nil {
			t.logger.Printf("bad glob %q: %v", in.Path, err)
			continue
		}
		for _, p := range paths {
			abs, err := filepath.Abs(p)
			if err != nil {
				continue
			}
			abs = filepath.ToSlash(abs)
			fi, err := os.Stat(abs)
			if err != nil || fi.IsDir() {
				continue
			}
			if in.IgnoreOlder.Duration > 0 && time.Since(fi.ModTime()) > in.IgnoreOlder.Duration {
				continue
			}
			found[abs] = in
		}
	}
	// Explicitly watched files (dynamic per user/session RAW.log tracking).
	for abs, in := range watched {
		fi, err := os.Stat(abs)
		if err != nil || fi.IsDir() {
			continue
		}
		found[abs] = in
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	for path, h := range t.harvesters {
		if _, ok := found[path]; !ok {
			h.stop()
			delete(t.harvesters, path)
		}
	}
	for path, in := range found {
		if _, ok := t.harvesters[path]; ok {
			continue
		}
		h := newHarvester(path, in, t.registry, t.hub, t.logger)
		t.harvesters[path] = h
		t.wg.Add(1)
		go h.run(&t.wg)
	}
}

// LookupSource resolves an input name to the file paths its glob currently
// matches (used by the read_cached_content endpoint). Includes dynamically
// watched files registered via WatchFile.
func (t *Tailer) LookupSource(name string) []string {
	var out []string
	t.mu.Lock()
	inputs := append([]config.InputConfig(nil), t.inputs...)
	watched := make(map[string]config.InputConfig, len(t.watched))
	for p, in := range t.watched {
		watched[p] = in
	}
	t.mu.Unlock()
	for _, in := range inputs {
		if in.Name != name {
			continue
		}
		paths, err := filepath.Glob(in.Path)
		if err != nil {
			continue
		}
		for _, p := range paths {
			abs, err := filepath.Abs(p)
			if err != nil {
				continue
			}
			abs = filepath.ToSlash(abs)
			if fi, err := os.Stat(abs); err == nil && !fi.IsDir() {
				out = append(out, abs)
			}
		}
	}
	for abs, in := range watched {
		if in.Name != name {
			continue
		}
		if fi, err := os.Stat(abs); err == nil && !fi.IsDir() {
			out = append(out, abs)
		}
	}
	return out
}

// SourceNames returns the distinct input names from static config plus
// dynamically registered source names (RegisterSourceName / WatchFile).
func (t *Tailer) SourceNames() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	seen := make(map[string]bool)
	var out []string
	for _, in := range t.inputs {
		if in.Name == "" || seen[in.Name] {
			continue
		}
		seen[in.Name] = true
		out = append(out, in.Name)
	}
	for name := range t.dynNames {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

// SourceOfPath returns the input name whose glob matches the given path, or
// the source name of a dynamically watched file, or "" if not covered.
func (t *Tailer) SourceOfPath(path string) string {
	abs := filepath.ToSlash(path)
	// Check dynamic watches first (exact match).
	t.mu.Lock()
	if in, ok := t.watched[abs]; ok && in.Name != "" {
		t.mu.Unlock()
		return in.Name
	}
	inputs := append([]config.InputConfig(nil), t.inputs...)
	t.mu.Unlock()
	for _, in := range inputs {
		paths, err := filepath.Glob(in.Path)
		if err != nil {
			continue
		}
		for _, p := range paths {
			pa, err := filepath.Abs(p)
			if err != nil {
				continue
			}
			if filepath.ToSlash(pa) == abs {
				return in.Name
			}
		}
	}
	return ""
}

func (t *Tailer) Stop() {
	close(t.stop)
	t.mu.Lock()
	for _, h := range t.harvesters {
		h.stop()
	}
	t.harvesters = make(map[string]*harvester)
	t.mu.Unlock()
	t.wg.Wait()
}

type harvester struct {
	path   string
	input  config.InputConfig
	reg    *Registry
	hub    *hub.Hub
	logger *log.Logger
	ml     *Multiline
	mode   string
	// mlTimeout is the dangling-buffer flush timeout for line/prefix modes.
	mlTimeout time.Duration

	f          *os.File
	buf        []byte // partial line(s) not yet terminated by '\n'
	offset     int64  // total bytes read from the file into buf
	openedInfo os.FileInfo

	stopCh chan struct{}
	once   sync.Once
}

func newHarvester(path string, in config.InputConfig, reg *Registry, h *hub.Hub, logger *log.Logger) *harvester {
	mode := in.Mode
	if mode == "" {
		mode = "line"
	}
	hv := &harvester{
		path:   path,
		input:  in,
		reg:    reg,
		hub:    h,
		logger: logger,
		mode:   mode,
		stopCh: make(chan struct{}),
	}

	switch mode {
	case "raw":
		// no line parsing; content is streamed as raw chunks
	case "prefix":
		re, err := strftimeToRegex(in.PrefixFormat)
		if err != nil {
			logger.Printf("input %q: invalid prefix_format %q: %v", in.Name, in.PrefixFormat, err)
		} else {
			hv.ml = NewMultilineFromRegex(re, false, "after")
		}
		hv.mlTimeout = in.PrefixTimeout.Duration
	default:
		ml, err := NewMultiline(in.Multiline.Pattern, in.Multiline.Negate, in.Multiline.Match)
		if err != nil {
			logger.Printf("input %q: invalid multiline pattern %q: %v", in.Name, in.Multiline.Pattern, err)
		} else {
			hv.ml = ml
		}
		hv.mlTimeout = in.Multiline.Timeout.Duration
	}
	return hv
}

func (h *harvester) stop() {
	h.once.Do(func() { close(h.stopCh) })
}

func (h *harvester) run(wg *sync.WaitGroup) {
	defer wg.Done()
	defer func() {
		if h.ml != nil {
			if msg, ok := h.ml.Flush(); ok {
				h.publish(msg, h.checkpoint())
			}
		}
		if h.f != nil {
			_ = h.f.Close()
		}
		h.reg.SetOffset(h.path, h.checkpoint())
	}()

	if err := h.open(); err != nil {
		h.logger.Printf("open %s: %v", h.path, err)
		return
	}

	if h.mode == "raw" {
		h.runRaw()
		return
	}
	h.runLines()
}

// runRaw streams the file content as-is: every read chunk is emitted as one
// base64-encoded event, with no line or prefix parsing.
func (h *harvester) runRaw() {
	scratch := make([]byte, rawChunkSize)
	for {
		n, err := h.f.Read(scratch)
		if n > 0 {
			h.offset += int64(n)
			h.publishRaw(scratch[:n], h.offset)
			continue
		}
		if err == nil {
			continue
		}
		switch {
		case isEofLike(err):
			h.reg.SetOffset(h.path, h.offset)
			if h.reopenIfRotated() {
				continue
			}
			select {
			case <-h.stopCh:
				return
			case <-time.After(h.input.PollInterval.Duration):
			}
		default:
			h.logger.Printf("read %s: %v", h.path, err)
			return
		}
	}
}

// runLines reads newline-delimited content, optionally grouping lines into
// multi-line events (line mode with multiline config, or prefix mode).
func (h *harvester) runLines() {
	lastActivity := time.Now()
	scratch := make([]byte, rawChunkSize)

	for {
		n, err := h.f.Read(scratch)
		if n > 0 {
			h.buf = append(h.buf, scratch[:n]...)
			h.offset += int64(n)
			for {
				idx := bytes.IndexByte(h.buf, '\n')
				if idx < 0 {
					break
				}
				line := h.buf[:idx]
				h.buf = h.buf[idx+1:]
				msg := strings.TrimSuffix(string(line), "\r")
				if out, ok := h.mlProcess(msg); ok {
					h.publish(out, h.checkpoint())
				}
			}
			if len(h.buf) > maxLineBytes {
				msg := strings.TrimSuffix(string(h.buf), "\r")
				h.buf = h.buf[:0]
				if out, ok := h.mlProcess(msg); ok {
					h.publish(out, h.checkpoint())
				}
			}
			lastActivity = time.Now()
			continue
		}

		if err == nil {
			continue
		}

		switch {
		case isEofLike(err):
			h.reg.SetOffset(h.path, h.checkpoint())

			if h.ml != nil && h.mlTimeout > 0 && time.Since(lastActivity) > h.mlTimeout {
				if msg, ok := h.ml.Flush(); ok {
					h.publish(msg, h.checkpoint())
				}
				lastActivity = time.Now()
			}

			if h.reopenIfRotated() {
				lastActivity = time.Now()
				continue
			}

			select {
			case <-h.stopCh:
				return
			case <-time.After(h.input.PollInterval.Duration):
			}
		default:
			h.logger.Printf("read %s: %v", h.path, err)
			return
		}
	}
}

// isEofLike reports whether a read error should be handled exactly like EOF:
// checkpoint and keep polling. Bind-mounted Windows directories (9p/DrvFS, as
// used by the Docker dev container) return ENODATA ("no data available")
// instead of io.EOF when reading at the end of a file; treating that as fatal
// killed the harvester permanently and froze the live stream.
func isEofLike(err error) bool {
	if errors.Is(err, io.EOF) {
		return true
	}
	return strings.Contains(err.Error(), "no data available")
}

func (h *harvester) open() error {
	fi, err := os.Stat(h.path)
	if err != nil {
		return err
	}

	start := h.reg.State(h.path)
	offset := start.Offset

	switch {
	case h.input.HasStartOffset:
		// Dynamically watched file: resume exactly from the registration
		// baseline (bytes between registration and attach must stream too).
		// Falls back to 0 when the file shrank (truncated/rotated in place).
		offset = h.input.StartOffset
		if offset > fi.Size() {
			offset = 0
		}
		h.logger.Printf("harvesting new file %s (mode=%s, offset=%d)", h.path, h.mode, offset)
	case !start.Seen:
		h.logger.Printf("harvesting new file %s (mode=%s)", h.path, h.mode)
		if h.input.TailFiles {
			offset = fi.Size()
		}
	case offset > fi.Size():
		h.logger.Printf("file %s shrank while stopped, restarting from beginning", h.path)
		offset = 0
	}

	f, err := os.Open(h.path)
	if err != nil {
		return err
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		_ = f.Close()
		return err
	}

	h.f = f
	h.buf = h.buf[:0]
	h.offset = offset
	h.openedInfo = fi
	h.reg.Update(h.path, offset, fi.Size(), fi.ModTime().UnixNano())
	return nil
}

// checkpoint returns the byte position just after the last complete line, so a
// restart resumes on a newline boundary (partial lines are re-read).
func (h *harvester) checkpoint() int64 {
	if int64(len(h.buf)) > h.offset {
		return 0
	}
	return h.offset - int64(len(h.buf))
}

// reopenIfRotated detects and handles the two common log rotation strategies:
// copy/truncate (same file, size shrinks) and rename (a new file replaces the
// path). Returns true when the file was reopened.
func (h *harvester) reopenIfRotated() bool {
	fi, err := os.Stat(h.path)
	if err != nil {
		return false
	}
	switch {
	case !os.SameFile(h.openedInfo, fi):
		h.logger.Printf("rotation detected for %s (new file), reopening", h.path)
	case fi.Size() < h.offset:
		h.logger.Printf("truncation detected for %s", h.path)
	default:
		return false
	}

	_ = h.f.Close()

	offset := int64(0)
	if h.input.TailFiles {
		offset = fi.Size()
	}
	nf, err := os.Open(h.path)
	if err != nil {
		return false
	}
	if _, err := nf.Seek(offset, io.SeekStart); err != nil {
		_ = nf.Close()
		return false
	}

	h.f = nf
	h.buf = h.buf[:0]
	h.offset = offset
	h.openedInfo = fi
	h.reg.Update(h.path, offset, fi.Size(), fi.ModTime().UnixNano())
	return true
}

func (h *harvester) mlProcess(line string) (string, bool) {
	if h.ml == nil {
		return line, true
	}
	return h.ml.Process(line)
}

func (h *harvester) publish(msg string, offset int64) {
	if filepath.Base(h.path) == "interaction.log" {
		msg = cleanInteractionLine(msg)
	}
	input := h.input.Name
	if input == "" {
		input = h.path
	}
	h.hub.Publish(hub.Event{
		Input:     input,
		File:      filepath.Base(h.path),
		Path:      h.path,
		Message:   msg,
		Mode:      h.mode,
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Offset:    offset,
		Session:   sessionFromAgentLogPath(h.path),
	})
}

func (h *harvester) publishRaw(chunk []byte, offset int64) {
	input := h.input.Name
	if input == "" {
		input = h.path
	}
	h.hub.Publish(hub.Event{
		Input:     input,
		File:      filepath.Base(h.path),
		Path:      h.path,
		Data:      base64.StdEncoding.EncodeToString(chunk),
		Encoding:  "base64",
		Mode:      h.mode,
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Offset:    offset,
		Session:   sessionFromAgentLogPath(h.path),
	})
}

// sessionFromAgentLogPath extracts the session id from per-session agent log
// paths, or "" for other files:
//
//	.../task_logs/<user>/<session>/RAW.log
//	.../<user>/<session>/run/.logs/interaction.log
func sessionFromAgentLogPath(path string) string {
	parts := strings.Split(filepath.ToSlash(path), "/")
	if len(parts) >= 3 && parts[len(parts)-1] == "RAW.log" {
		return parts[len(parts)-2]
	}
	if len(parts) >= 4 && parts[len(parts)-1] == "interaction.log" && parts[len(parts)-2] == ".logs" {
		return parts[len(parts)-4]
	}
	return ""
}

var (
	ansiEscapeRe   = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)
	loguruPrefixRe = regexp.MustCompile(`^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+\s*\| `)
)

// cleanInteractionLine normalizes one interaction.log line for SSE consumers:
// ANSI color escapes and the loguru timestamp/level prefix are stripped so the
// payload ([user] ..., [tool-call] ..., [assistant] ..., tool output) stays.
func cleanInteractionLine(line string) string {
	return loguruPrefixRe.ReplaceAllString(ansiEscapeRe.ReplaceAllString(line, ""), "")
}
