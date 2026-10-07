package hub

import (
	"encoding/json"
	"sync"
	"sync/atomic"
)

// Event is a single log record streamed to SSE consumers.
type Event struct {
	Input     string `json:"input"`
	File      string `json:"file"`
	Path      string `json:"path"`
	Message   string `json:"message,omitempty"`  // text content (line / prefix modes)
	Data      string `json:"data,omitempty"`     // base64 content (raw mode)
	Encoding  string `json:"encoding,omitempty"` // "base64" when Data is set
	Mode      string `json:"mode,omitempty"`     // line | raw | prefix
	Timestamp string `json:"timestamp"`
	Offset    int64  `json:"offset"`
	// Session identifies the owning agent session for per-session agent logs
	// (.../task_logs/<user>/<session>/RAW.log). Empty for static sources.
	// Session-scoped SSE consumers filter on this to avoid cross-session leaks.
	Session string `json:"session,omitempty"`
}

func (e Event) Marshal() []byte {
	b, _ := json.Marshal(e)
	return b
}

type Subscriber struct {
	id     int64
	ch     chan Event
	filter atomic.Pointer[map[string]bool]
}

// Events returns the receive-only channel of events for this subscriber.
func (s *Subscriber) Events() <-chan Event {
	return s.ch
}

// SetSources updates the source filter for this subscriber. A nil slice means
// "match every source"; an empty slice means "match nothing".
func (s *Subscriber) SetSources(sources []string) {
	if sources == nil {
		s.filter.Store(nil)
		return
	}
	m := make(map[string]bool, len(sources))
	for _, src := range sources {
		m[src] = true
	}
	s.filter.Store(&m)
}

// accepts reports whether the event matches the subscriber's source filter.
func (s *Subscriber) accepts(e Event) bool {
	m := s.filter.Load()
	if m == nil {
		return true
	}
	return (*m)[e.Input] || (*m)[e.Path]
}

// Hub fans out events to connected SSE subscribers.
type Hub struct {
	mu   sync.RWMutex
	subs map[*Subscriber]struct{}
	next int64
}

func New() *Hub {
	return &Hub{subs: make(map[*Subscriber]struct{})}
}

func (h *Hub) Subscribe(buffer int) *Subscriber {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := &Subscriber{id: h.next, ch: make(chan Event, buffer)}
	h.next++
	h.subs[s] = struct{}{}
	return s
}

func (h *Hub) Unsubscribe(s *Subscriber) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.subs[s]; ok {
		delete(h.subs, s)
		close(s.ch)
	}
}

func (h *Hub) SubscriberCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.subs)
}

// Publish delivers the event to every matching subscriber without blocking: a
// slow consumer has its backlog dropped rather than stalling log tailing.
func (h *Hub) Publish(e Event) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for s := range h.subs {
		if !s.accepts(e) {
			continue
		}
		select {
		case s.ch <- e:
		default:
		}
	}
}
