package server

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"log_gateway_server/internal/agent"
	"log_gateway_server/internal/auth"
	"log_gateway_server/internal/config"
	"log_gateway_server/internal/hub"
	"log_gateway_server/internal/session"
)

// newRestoreTestEnv builds a server whose agent runner points at a temp
// workspace root pre-populated with historical session directories.
func newRestoreTestEnv(t *testing.T, root string) (*usersTestEnv, string) {
	t.Helper()
	authPath := filepath.Join(t.TempDir(), "AUTH.yaml")
	if err := os.WriteFile(authPath, []byte("boss:\n  password: b\n  role: admin\nalice: a\n"), 0o600); err != nil {
		t.Fatalf("write auth: %v", err)
	}
	store, err := auth.Load(authPath, []byte("secret"))
	if err != nil {
		t.Fatalf("load auth: %v", err)
	}
	logger := log.New(io.Discard, "", 0)
	runner := agent.New(config.AgentConfig{WorkspaceRoot: root, LogDirName: "task_logs"}, logger)
	srv := New(
		hub.New(), "127.0.0.1", 0, "/events", logger, store,
		session.NewManager(time.Hour), time.Hour, "",
		func() []string { return []string{"agent-logs"} },
		func(string) string { return "" },
		func(string) []string { return nil },
		runner, nil, config.WorkspaceConfig{},
	)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return &usersTestEnv{ts: ts, authPath: authPath, srv: srv}, root
}

func TestRestoreSessionsEndToEnd(t *testing.T) {
	root := t.TempDir()
	// Historical session for alice: query file + RAW.log (lite task).
	sessDir := filepath.Join(root, "alice", "sess-1")
	if err := os.MkdirAll(sessDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(sessDir, agent.FirstUserQueryFile), []byte("historic question"), 0o644); err != nil {
		t.Fatalf("write query: %v", err)
	}
	rawDir := filepath.Join(root, "task_logs", "alice", "sess-1")
	if err := os.MkdirAll(rawDir, 0o755); err != nil {
		t.Fatalf("mkdir raw: %v", err)
	}
	if err := os.WriteFile(filepath.Join(rawDir, "RAW.log"), []byte("old transcript\n"), 0o644); err != nil {
		t.Fatalf("write raw: %v", err)
	}
	// Empty shell must not be restored.
	if err := os.MkdirAll(filepath.Join(root, "alice", "empty"), 0o755); err != nil {
		t.Fatalf("mkdir empty: %v", err)
	}

	env, _ := newRestoreTestEnv(t, root)

	if n := env.srv.RestoreSessions(); n != 1 {
		t.Fatalf("RestoreSessions = %d, want 1", n)
	}
	if n := env.srv.RestoreSessions(); n != 0 {
		t.Errorf("second RestoreSessions = %d, want 0 (already registered)", n)
	}

	// alice logs in and sees the historical session with its name.
	code, login := loginJSON(t, env.ts, "alice", "a")
	if code != http.StatusOK {
		t.Fatalf("login = %d", code)
	}
	tok := login["token"].(string)
	resp, body := doJSON(t, http.MethodGet, env.ts.URL+"/sessions", tok, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("list sessions = %d", resp.StatusCode)
	}
	sessions := body["sessions"].([]any)
	if len(sessions) != 1 {
		t.Fatalf("sessions = %v, want 1 restored entry", sessions)
	}
	first := sessions[0].(map[string]any)
	if first["session_id"] != "sess-1" || first["name"] != "historic question" {
		t.Errorf("restored session = %v", first)
	}
	if first["last_active"] == "" || first["last_active"] == nil {
		t.Errorf("last_active missing: %v", first)
	}

	// The stored id must resolve again: activate works and keeps sources.
	resp, acc := doJSON(t, http.MethodPost, env.ts.URL+"/sessions/sess-1/activate", tok, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("activate = %d", resp.StatusCode)
	}
	if acc["session_id"] != "sess-1" || acc["user"] != "alice" {
		t.Errorf("activate body = %v", acc)
	}
	srcs, _ := acc["sources"].([]any)
	if len(srcs) != 1 || srcs[0] != "agent-logs" {
		t.Errorf("restored sources = %v", acc["sources"])
	}

	// Other users cannot see or activate alice's restored session.
	code, login = loginJSON(t, env.ts, "boss", "b")
	if code != http.StatusOK {
		t.Fatalf("boss login = %d", code)
	}
	bossTok := login["token"].(string)
	_, body = doJSON(t, http.MethodGet, env.ts.URL+"/sessions", bossTok, nil)
	if got := body["sessions"].([]any); len(got) != 0 {
		t.Errorf("boss sessions = %v, want none", got)
	}
	resp, _ = doJSON(t, http.MethodPost, env.ts.URL+"/sessions/sess-1/activate", bossTok, nil)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("boss activate alice session = %d, want 401", resp.StatusCode)
	}
}
