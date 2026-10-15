package agent

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeFileAt(t *testing.T, path, content string, mtime time.Time) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
}

func TestDiscoverSessions(t *testing.T) {
	root := t.TempDir()
	t1 := time.Date(2026, 10, 1, 10, 0, 0, 0, time.UTC)
	t2 := time.Date(2026, 10, 2, 10, 0, 0, 0, time.UTC)
	t3 := time.Date(2026, 10, 3, 10, 0, 0, 0, time.UTC)

	// Lite session: query file drives the name (first line, collapsed).
	writeFileAt(t, filepath.Join(root, "alice", "sess-a", FirstUserQueryFile),
		"  first   question\nsecond line ", t1)
	// Heavy session: manifest drives the name; the newer RAW.log drives the time.
	writeFileAt(t, filepath.Join(root, "alice", "sess-b", HeavyManifestName),
		"defaults:\n  type: lnr\ntasks:\n  - exp_id: run\n    run_id: sess-b\n    task: \"heavy task\"\n", t2)
	writeFileAt(t, filepath.Join(root, "task_logs", "alice", "sess-b", "RAW.log"), "log", t3)
	// Interaction-log-only session: restored without a name.
	writeFileAt(t, filepath.Join(root, "bob", "sess-c", RunDirName, ".logs", "interaction.log"), "x", t1.Add(-time.Hour))
	// Empty shell (no task artifacts): skipped.
	if err := os.MkdirAll(filepath.Join(root, "alice", "empty"), 0o755); err != nil {
		t.Fatalf("mkdir empty: %v", err)
	}
	// task_logs user directory and stray files are skipped.

	got, err := DiscoverSessions(root)
	if err != nil {
		t.Fatalf("DiscoverSessions: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("sessions = %d (%+v), want 3", len(got), got)
	}
	// Newest first: sess-b (t3) > sess-a (t1) > sess-c.
	if got[0].ID != "sess-b" || got[0].User != "alice" || got[0].Name != "heavy task" {
		t.Errorf("got[0] = %+v", got[0])
	}
	if !got[0].LastActive.Equal(t3) {
		t.Errorf("sess-b last = %v, want %v", got[0].LastActive, t3)
	}
	if got[1].ID != "sess-a" || got[1].Name != "first question" {
		t.Errorf("got[1] = %+v", got[1])
	}
	if !got[1].LastActive.Equal(t1) {
		t.Errorf("sess-a last = %v, want %v", got[1].LastActive, t1)
	}
	if got[2].ID != "sess-c" || got[2].User != "bob" || got[2].Name != "" {
		t.Errorf("got[2] = %+v", got[2])
	}
}

func TestSessionNameFromManifestLite(t *testing.T) {
	b := []byte("defaults:\n  repl_profile: lite\ntasks:\n  - exp_id: run\n    run_id: s\n    task: \"lite question\"\n")
	if got := sessionNameFromManifest(b); got != "lite question" {
		t.Errorf("name = %q, want %q", got, "lite question")
	}
	if got := sessionNameFromManifest([]byte("not: [valid")); got != "" {
		t.Errorf("invalid manifest name = %q, want empty", got)
	}
}
