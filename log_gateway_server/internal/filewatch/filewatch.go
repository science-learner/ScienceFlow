package filewatch

import (
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Snapshot limits keep a single scan lightweight even for big workspaces.
const (
	maxDepth        = 10
	maxTotalEntries = 5000
	maxChanges      = 2000
)

// Node is one entry of a workspace file tree.
type Node struct {
	Name  string `json:"name"`
	Path  string `json:"path"` // slash-separated path relative to the workspace root
	Type  string `json:"type"` // file | dir | symlink
	Size  int64  `json:"size"`
	MTime int64  `json:"mtime"` // unix nanoseconds
}

// Event is pushed to SSE subscribers: either the initial full tree or an
// incremental diff since the previous poll.
type Event struct {
	Kind      string `json:"kind"` // "tree" | "changes"
	Root      string `json:"root,omitempty"`
	Tree      []Node `json:"tree,omitempty"`
	Added     []Node `json:"added,omitempty"`
	Removed   []Node `json:"removed,omitempty"`
	Modified  []Node `json:"modified,omitempty"`
	Overflow  bool   `json:"overflow,omitempty"` // true when a diff was capped; frontend should refetch the tree
	Timestamp string `json:"timestamp"`
}

// Snapshot walks root and returns a sorted, bounded file tree. Missing or
// unreadable directories yield an empty tree. VCS internals are skipped and
// symlinks are reported (not followed), so huge linked datasets stay cheap.
func Snapshot(root string) []Node {
	nodes := []Node{}
	total := 0
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == ".hg" || d.Name() == ".svn") {
			return fs.SkipDir
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		if rel == "." {
			return nil
		}
		depth := strings.Count(rel, string(filepath.Separator)) + 1
		if depth > maxDepth {
			if d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if total >= maxTotalEntries {
			return fs.SkipAll
		}
		total++
		info, err := d.Info()
		if err != nil {
			return nil
		}
		typ := "file"
		switch {
		case d.IsDir():
			typ = "dir"
		case d.Type()&fs.ModeSymlink != 0:
			typ = "symlink"
		}
		nodes = append(nodes, Node{
			Name:  d.Name(),
			Path:  filepath.ToSlash(rel),
			Type:  typ,
			Size:  info.Size(),
			MTime: info.ModTime().UnixNano(),
		})
		return nil
	})
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Path < nodes[j].Path })
	return nodes
}

// meta is the change-relevant subset of a node used for diffing.
type meta struct {
	size  int64
	mtime int64
	typ   string
}

func nodeFrom(path string, m meta) Node {
	return Node{Name: filepath.Base(path), Path: path, Type: m.typ, Size: m.size, MTime: m.mtime}
}

type watcher struct {
	root     string
	interval time.Duration

	mu      sync.Mutex
	subs    map[int]chan Event
	nextID  int
	last    map[string]meta
	stopped bool
	stop    chan struct{}
}

func newWatcher(root string, interval time.Duration) *watcher {
	return &watcher{
		root:     root,
		interval: interval,
		subs:     make(map[int]chan Event),
		stop:     make(chan struct{}),
	}
}

func (w *watcher) isStopped() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stopped
}

func (w *watcher) addSub() (int, chan Event) {
	ch := make(chan Event, 16)
	w.mu.Lock()
	id := w.nextID
	w.nextID++
	w.subs[id] = ch
	w.mu.Unlock()
	return id, ch
}

// removeSub detaches a subscriber and reports whether the watcher should stop.
func (w *watcher) removeSub(id int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if ch, ok := w.subs[id]; ok {
		delete(w.subs, id)
		close(ch)
	}
	if len(w.subs) == 0 && !w.stopped {
		w.stopped = true
		close(w.stop)
		return true
	}
	return false
}

func (w *watcher) snapshotMap() map[string]meta {
	m := make(map[string]meta)
	for _, n := range Snapshot(w.root) {
		m[n.Path] = meta{size: n.Size, mtime: n.MTime, typ: n.Type}
	}
	return m
}

func (w *watcher) run(done func(root string)) {
	defer done(w.root)
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	for {
		select {
		case <-w.stop:
			return
		case <-ticker.C:
			w.poll()
		}
	}
}

func (w *watcher) poll() {
	cur := w.snapshotMap()
	w.mu.Lock()
	prev := w.last
	w.last = cur
	subs := make([]chan Event, 0, len(w.subs))
	for _, ch := range w.subs {
		subs = append(subs, ch)
	}
	w.mu.Unlock()
	if prev == nil || len(subs) == 0 {
		// First poll only establishes the baseline; the initial tree was
		// already delivered at subscribe time.
		return
	}

	var added, removed, modified []Node
	for p, m := range cur {
		if om, ok := prev[p]; !ok {
			added = append(added, nodeFrom(p, m))
		} else if om != m {
			modified = append(modified, nodeFrom(p, m))
		}
	}
	for p, m := range prev {
		if _, ok := cur[p]; !ok {
			removed = append(removed, nodeFrom(p, m))
		}
	}
	if len(added) == 0 && len(removed) == 0 && len(modified) == 0 {
		return
	}

	sortNodes(added)
	sortNodes(removed)
	sortNodes(modified)

	overflow := false
	if len(added) > maxChanges {
		added = added[:maxChanges]
		overflow = true
	}
	if len(removed) > maxChanges {
		removed = removed[:maxChanges]
		overflow = true
	}
	if len(modified) > maxChanges {
		modified = modified[:maxChanges]
		overflow = true
	}

	ev := Event{
		Kind:      "changes",
		Added:     added,
		Removed:   removed,
		Modified:  modified,
		Overflow:  overflow,
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
	}
	for _, ch := range subs {
		select {
		case ch <- ev:
		default: // slow consumer: dropped; recover via full tree refetch
		}
	}
}

func sortNodes(nodes []Node) {
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].Path < nodes[j].Path })
}

// Manager owns one watcher goroutine per active workspace root. Watchers run
// only while at least one subscriber (SSE connection) is attached.
type Manager struct {
	mu       sync.Mutex
	ws       map[string]*watcher
	interval time.Duration
}

func New(interval time.Duration) *Manager {
	if interval <= 0 {
		interval = 3 * time.Second
	}
	return &Manager{ws: make(map[string]*watcher), interval: interval}
}

// Subscribe registers a consumer for a workspace root. It immediately delivers
// the current full tree, then periodic diff events. The returned cancel func
// detaches the consumer; the watcher goroutine stops when no consumers remain.
func (m *Manager) Subscribe(root string) (<-chan Event, func()) {
	root = filepath.ToSlash(root)
	m.mu.Lock()
	w, ok := m.ws[root]
	if !ok || w.isStopped() {
		w = newWatcher(root, m.interval)
		m.ws[root] = w
		go w.run(m.watcherDone)
	}
	id, ch := w.addSub()
	m.mu.Unlock()

	ev := Event{
		Kind:      "tree",
		Root:      root,
		Tree:      Snapshot(root),
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
	}
	select {
	case ch <- ev:
	default:
	}

	return ch, func() {
		if w.removeSub(id) {
			m.watcherDone(root)
		}
	}
}

// watcherDone removes a stopped watcher from the manager map. Idempotent.
func (m *Manager) watcherDone(root string) {
	m.mu.Lock()
	if w, ok := m.ws[root]; ok {
		w.mu.Lock()
		empty := len(w.subs) == 0
		w.mu.Unlock()
		if empty {
			delete(m.ws, root)
		}
	}
	m.mu.Unlock()
}
