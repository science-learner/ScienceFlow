package session

import (
	"crypto/rand"
	"encoding/hex"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Session is one client's subscription: a set of sources it currently follows.
type Session struct {
	id       string
	name     string // human-readable label; empty until named after the first user question
	named    bool   // true once the name has been derived from the first user question
	user     string
	mu       sync.RWMutex
	srcs     map[string]bool
	last     atomic.Int64  // last activity of any kind (SSE, reads, ...), unix nanoseconds; drives idle TTL
	chatLast atomic.Int64  // last user chat action (invoke/stop), unix nanoseconds; drives "last_active" display & ordering
	notify   chan struct{} // capacity 1: signals a source-set change
}

// Manager holds in-memory sessions with idle expiry, safe for concurrent use.
type Manager struct {
	mu       sync.Mutex
	sessions map[string]*Session
	ttl      time.Duration
	stop     chan struct{}
	once     sync.Once
}

func NewManager(ttl time.Duration) *Manager {
	m := &Manager{
		sessions: make(map[string]*Session),
		ttl:      ttl,
		stop:     make(chan struct{}),
	}
	go m.cleanup()
	return m
}

func newID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func (m *Manager) Create(user string, sources []string) string {
	id := newID()
	s := &Session{
		id:     id,
		name:   "", // named after the first user question (MaybeNameFromQuery)
		user:   user,
		srcs:   make(map[string]bool),
		notify: make(chan struct{}, 1),
	}
	s.last.Store(time.Now().UnixNano())
	s.chatLast.Store(time.Now().UnixNano()) // a fresh session counts as its own last chat time
	s.setSources(sources)

	m.mu.Lock()
	m.sessions[id] = s
	m.mu.Unlock()
	return id
}

func (m *Manager) Get(id string) *Session {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sessions[id]
	if !ok {
		return nil
	}
	s.last.Store(time.Now().UnixNano())
	return s
}

func (m *Manager) Delete(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[id]; !ok {
		return false
	}
	delete(m.sessions, id)
	return true
}

// SetSources replaces the session's source set and notifies any listener.
func (m *Manager) SetSources(id string, sources []string) bool {
	s := m.Get(id)
	if s == nil {
		return false
	}
	s.setSources(sources)
	return true
}

// Sources returns a copy of the session's current source set.
func (m *Manager) Sources(id string) []string {
	s := m.Get(id)
	if s == nil {
		return nil
	}
	return s.sources()
}

// User returns the session owner's username.
func (m *Manager) User(id string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok {
		return s.user
	}
	return ""
}

// Name returns the session's human-readable label, or "" if absent.
func (m *Manager) Name(id string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok {
		return s.name
	}
	return ""
}

// ListByUser returns the live sessions owned by user, most recently chatted
// first. Touching is left to the caller so a pure read does not skew TTLs.
func (m *Manager) ListByUser(user string) []*Session {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*Session, 0, len(m.sessions))
	for _, s := range m.sessions {
		if s.user == user {
			out = append(out, s)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].chatLast.Load() > out[j].chatLast.Load()
	})
	return out
}

// TouchChat stamps the session's last user chat action (invoke/stop) without
// disturbing the idle-TTL clock.
func (m *Manager) TouchChat(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok {
		s.chatLast.Store(time.Now().UnixNano())
	}
}

// MaybeNameFromQuery names the session after its first user question: the
// first line of the query, whitespace-collapsed. Takes effect only once —
// later questions never rename the session. Empty queries are ignored.
func (m *Manager) MaybeNameFromQuery(id string, query string) {
	line := strings.TrimSpace(query)
	if i := strings.IndexAny(line, "\r\n"); i >= 0 {
		line = line[:i]
	}
	line = strings.Join(strings.Fields(line), " ")
	if line == "" {
		return
	}
	if runes := []rune(line); len(runes) > 120 { // storage cap; display truncation is the frontend's job
		line = string(runes[:120])
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sessions[id]; ok && !s.named {
		s.name = line
		s.named = true
	}
}

// Notify returns the session's change-notification channel.
func (m *Manager) Notify(id string) <-chan struct{} {
	s := m.Get(id)
	if s == nil {
		return nil
	}
	return s.notify
}

func (m *Manager) cleanup() {
	t := time.NewTicker(time.Minute)
	defer t.Stop()
	for {
		select {
		case <-m.stop:
			return
		case <-t.C:
			now := time.Now().UnixNano()
			expire := int64(m.ttl)
			m.mu.Lock()
			for id, s := range m.sessions {
				if now-s.last.Load() > expire {
					delete(m.sessions, id)
				}
			}
			m.mu.Unlock()
		}
	}
}

// Stop terminates the cleanup goroutine (called on shutdown).
func (m *Manager) Stop() {
	m.once.Do(func() { close(m.stop) })
}

func (s *Session) setSources(sources []string) {
	s.mu.Lock()
	s.srcs = make(map[string]bool, len(sources))
	for _, src := range sources {
		s.srcs[src] = true
	}
	s.mu.Unlock()
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

func (s *Session) sources() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]string, 0, len(s.srcs))
	for src := range s.srcs {
		out = append(out, src)
	}
	sort.Strings(out)
	return out
}

// ID returns the session identifier.
func (s *Session) ID() string { return s.id }

// Name returns the human-readable session label.
func (s *Session) Name() string { return s.name }

// User returns the session owner's username.
func (s *Session) User() string { return s.user }

// Sources returns a copy of the session's source set.
func (s *Session) Sources() []string { return s.sources() }

// LastActive returns the last activity timestamp (any kind; drives idle TTL).
func (s *Session) LastActive() time.Time { return time.Unix(0, s.last.Load()) }

// ChatActive returns the last user chat action timestamp (invoke/stop).
func (s *Session) ChatActive() time.Time { return time.Unix(0, s.chatLast.Load()) }
