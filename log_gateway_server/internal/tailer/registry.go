package tailer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// FileState is the persisted checkpoint for a single log file, mirroring the
// role of Filebeat's registry: it remembers the offset so a restart (or a
// harvester that was closed when a file disappeared) resumes where it left off.
type FileState struct {
	Seen    bool  `json:"seen"`
	Offset  int64 `json:"offset"`
	Size    int64 `json:"size"`
	ModTime int64 `json:"mod_time"`
}

type Registry struct {
	mu            sync.Mutex
	states        map[string]*FileState
	path          string
	flushInterval time.Duration

	stop chan struct{}
	wg   sync.WaitGroup
}

func NewRegistry(path string, flushInterval time.Duration) (*Registry, error) {
	r := &Registry{
		states:        make(map[string]*FileState),
		path:          path,
		flushInterval: flushInterval,
		stop:          make(chan struct{}),
	}
	if err := r.load(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *Registry) load() error {
	b, err := os.ReadFile(r.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if len(b) == 0 {
		return nil
	}
	var states map[string]*FileState
	if err := json.Unmarshal(b, &states); err != nil {
		return err
	}
	r.states = states
	return nil
}

// State returns the current checkpoint for path, creating an empty one if
// this file has never been seen before.
func (r *Registry) State(path string) *FileState {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.states[path]
	if !ok {
		s = &FileState{}
		r.states[path] = s
	}
	return s
}

// Update replaces the checkpoint and marks the file as seen.
func (r *Registry) Update(path string, offset, size, modTime int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.states[path] = &FileState{Seen: true, Offset: offset, Size: size, ModTime: modTime}
}

func (r *Registry) SetOffset(path string, offset int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.states[path]
	if !ok {
		s = &FileState{}
		r.states[path] = s
	}
	s.Offset = offset
}

func (r *Registry) Remove(path string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.states, path)
}

func (r *Registry) Flush() error {
	r.mu.Lock()
	b, err := json.MarshalIndent(r.states, "", "  ")
	r.mu.Unlock()
	if err != nil {
		return err
	}
	dir := filepath.Dir(r.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp := r.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	if _, err := os.Stat(r.path); err == nil {
		_ = os.Remove(r.path)
	}
	return os.Rename(tmp, r.path)
}

func (r *Registry) Start() {
	r.wg.Add(1)
	go r.loop()
}

func (r *Registry) loop() {
	defer r.wg.Done()
	t := time.NewTicker(r.flushInterval)
	defer t.Stop()
	for {
		select {
		case <-r.stop:
			return
		case <-t.C:
			_ = r.Flush()
		}
	}
}

func (r *Registry) Stop() {
	close(r.stop)
	r.wg.Wait()
	_ = r.Flush()
}
