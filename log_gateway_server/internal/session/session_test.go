package session

import (
	"testing"
	"time"
)

func TestRestore(t *testing.T) {
	m := NewManager(time.Hour)
	defer m.Stop()

	when := time.Date(2026, 10, 1, 8, 0, 0, 0, time.UTC)
	if !m.Restore("s1", "alice", "first question", []string{"agent-logs"}, when) {
		t.Fatal("Restore should succeed")
	}
	if m.Restore("s1", "alice", "dup", nil, when) {
		t.Error("duplicate restore must be rejected")
	}
	if m.Restore("", "alice", "", nil, when) || m.Restore("s2", "", "", nil, when) {
		t.Error("empty id/user must be rejected")
	}

	if m.User("s1") != "alice" {
		t.Errorf("User = %q", m.User("s1"))
	}
	if m.Name("s1") != "first question" {
		t.Errorf("Name = %q", m.Name("s1"))
	}
	if srcs := m.Sources("s1"); len(srcs) != 1 || srcs[0] != "agent-logs" {
		t.Errorf("Sources = %v", srcs)
	}
	if got := m.Get("s1").ChatActive(); !got.Equal(when) {
		t.Errorf("ChatActive = %v, want %v", got, when)
	}

	// A restored, already-named session is not renamed by later questions.
	m.MaybeNameFromQuery("s1", "another question")
	if m.Name("s1") != "first question" {
		t.Errorf("restored name changed: %q", m.Name("s1"))
	}

	// Unnamed restored session gets named by the next question.
	if !m.Restore("s2", "bob", "", nil, when) {
		t.Fatal("Restore s2")
	}
	m.MaybeNameFromQuery("s2", "new chat")
	if m.Name("s2") != "new chat" {
		t.Errorf("s2 name = %q", m.Name("s2"))
	}

	if list := m.ListByUser("alice"); len(list) != 1 || list[0].ID() != "s1" {
		t.Errorf("ListByUser(alice) = %v", list)
	}
}
