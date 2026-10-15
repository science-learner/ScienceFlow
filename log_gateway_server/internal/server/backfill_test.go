package server

import (
	"io"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	"log_gateway_server/internal/agent"
	"log_gateway_server/internal/config"
	"log_gateway_server/internal/hub"
	"log_gateway_server/internal/session"
)

// TestBackfillFilesIncludesHeavyLog guards the heavy-mode history replay:
// reconnecting to a session must backfill the LNR worker transcript
// (parallel_subprocess.log), not just RAW.log / interaction.log.
func TestBackfillFilesIncludesHeavyLog(t *testing.T) {
	root := t.TempDir()
	sessDir := filepath.Join(root, "alice", "sess-1")
	heavy := filepath.Join(sessDir, "run", "task_logs", agent.HeavySubprocessLogName)
	if err := os.MkdirAll(filepath.Dir(heavy), 0o755); err != nil {
		t.Fatalf("mkdir heavy dir: %v", err)
	}
	if err := os.WriteFile(heavy, []byte("worker transcript\n"), 0o644); err != nil {
		t.Fatalf("write heavy log: %v", err)
	}
	raw := filepath.Join(root, "task_logs", "alice", "sess-1", "RAW.log")
	if err := os.MkdirAll(filepath.Dir(raw), 0o755); err != nil {
		t.Fatalf("mkdir raw dir: %v", err)
	}
	if err := os.WriteFile(raw, []byte("raw log\n"), 0o644); err != nil {
		t.Fatalf("write raw log: %v", err)
	}

	logger := log.New(io.Discard, "", 0)
	runner := agent.New(config.AgentConfig{WorkspaceRoot: root, LogDirName: "task_logs"}, logger)
	sessions := session.NewManager(time.Hour)
	defer sessions.Stop()
	if !sessions.Restore("sess-1", "alice", "n", []string{"agent-logs"}, time.Now()) {
		t.Fatal("restore test session failed")
	}
	srv := New(
		hub.New(), "127.0.0.1", 0, "/events", logger, nil,
		sessions, time.Hour, "",
		func() []string { return []string{"agent-logs"} },
		func(string) string { return "" },
		func(string) []string { return nil },
		runner, nil, config.WorkspaceConfig{},
	)

	got := srv.backfillFiles("sess-1", agent.RawLogSource)
	want := []string{raw, heavy} // interaction.log absent: RAW.log then heavy worker log
	if len(got) != len(want) {
		t.Fatalf("backfillFiles = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("backfillFiles[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
