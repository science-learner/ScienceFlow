package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"log_gateway_server/internal/auth"
	"log_gateway_server/internal/config"
	"log_gateway_server/internal/hub"
	"log_gateway_server/internal/session"
)

type usersTestEnv struct {
	ts       *httptest.Server
	authPath string
	srv      *Server
}

func newUsersTestEnv(t *testing.T, authYAML string) *usersTestEnv {
	t.Helper()
	authPath := filepath.Join(t.TempDir(), "AUTH.yaml")
	if err := os.WriteFile(authPath, []byte(authYAML), 0o600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}
	store, err := auth.Load(authPath, []byte("test-secret"))
	if err != nil {
		t.Fatalf("load auth: %v", err)
	}
	srv := New(
		hub.New(), "127.0.0.1", 0, "/events",
		log.New(io.Discard, "", 0),
		store,
		session.NewManager(time.Hour),
		time.Hour,
		"",
		func() []string { return []string{"src-a"} },
		func(string) string { return "" },
		func(string) []string { return nil },
		nil, nil, config.WorkspaceConfig{},
	)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return &usersTestEnv{ts: ts, authPath: authPath, srv: srv}
}

func loginJSON(t *testing.T, ts *httptest.Server, user, pass string) (int, map[string]any) {
	t.Helper()
	body, _ := json.Marshal(map[string]string{"user_name": user, "password": pass})
	resp, err := http.Post(ts.URL+"/login", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("login request: %v", err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func doJSON(t *testing.T, method, url, token string, payload any) (*http.Response, map[string]any) {
	t.Helper()
	var body io.Reader
	if payload != nil {
		b, _ := json.Marshal(payload)
		body = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	defer resp.Body.Close()
	var out map[string]any
	raw, _ := io.ReadAll(resp.Body)
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &out)
	}
	return resp, out
}

func TestLoginIncludesRole(t *testing.T) {
	env := newUsersTestEnv(t, "boss:\n  password: b\n  role: admin\npleb: p\n")

	if code, body := loginJSON(t, env.ts, "boss", "b"); code != http.StatusOK || body["role"] != "admin" {
		t.Fatalf("admin login = %d %v", code, body)
	}
	if code, body := loginJSON(t, env.ts, "pleb", "p"); code != http.StatusOK || body["role"] != "user" {
		t.Fatalf("user login = %d %v", code, body)
	}
}

func TestUsersAPIRequiresAdmin(t *testing.T) {
	env := newUsersTestEnv(t, "boss:\n  password: b\n  role: admin\npleb: p\n")

	// No token -> 401.
	resp, _ := doJSON(t, http.MethodGet, env.ts.URL+"/users", "", nil)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("anonymous /users = %d, want 401", resp.StatusCode)
	}

	// Normal user -> 403.
	_, login := loginJSON(t, env.ts, "pleb", "p")
	tok := login["token"].(string)
	resp, _ = doJSON(t, http.MethodGet, env.ts.URL+"/users", tok, nil)
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("non-admin /users = %d, want 403", resp.StatusCode)
	}

	// Admin -> 200 with accounts.
	_, login = loginJSON(t, env.ts, "boss", "b")
	tok = login["token"].(string)
	resp, body := doJSON(t, http.MethodGet, env.ts.URL+"/users", tok, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("admin /users = %d", resp.StatusCode)
	}
	users := body["users"].([]any)
	if len(users) != 2 {
		t.Errorf("users = %v, want 2 entries", users)
	}
}

func TestUsersCRUDLifecycle(t *testing.T) {
	env := newUsersTestEnv(t, "boss:\n  password: b\n  role: admin\n")
	_, login := loginJSON(t, env.ts, "boss", "b")
	adminTok := login["token"].(string)

	// Create.
	resp, body := doJSON(t, http.MethodPost, env.ts.URL+"/users", adminTok, map[string]any{
		"name": "alice", "password": "pw1", "sources": []string{"src-a"},
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create = %d %v", resp.StatusCode, body)
	}
	// Login as the new user.
	code, aliceLogin := loginJSON(t, env.ts, "alice", "pw1")
	if code != http.StatusOK {
		t.Fatalf("alice login = %d", code)
	}
	aliceTok := aliceLogin["token"].(string)

	// Duplicate -> 409.
	resp, _ = doJSON(t, http.MethodPost, env.ts.URL+"/users", adminTok, map[string]any{"name": "alice", "password": "x"})
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("duplicate create = %d, want 409", resp.StatusCode)
	}
	// Invalid name -> 400.
	resp, _ = doJSON(t, http.MethodPost, env.ts.URL+"/users", adminTok, map[string]any{"name": "bad name", "password": "x"})
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("invalid name = %d, want 400", resp.StatusCode)
	}

	// Update password -> old token invalidated immediately (hot update).
	resp, _ = doJSON(t, http.MethodPut, env.ts.URL+"/users/alice", adminTok, map[string]any{"password": "pw2"})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("update = %d", resp.StatusCode)
	}
	respNamed, _ := doJSON(t, http.MethodGet, env.ts.URL+"/sources", aliceTok, nil)
	if respNamed.StatusCode != http.StatusUnauthorized {
		t.Errorf("old alice token after password change = %d, want 401", respNamed.StatusCode)
	}
	if code, _ := loginJSON(t, env.ts, "alice", "pw2"); code != http.StatusOK {
		t.Errorf("alice new password login = %d", code)
	}

	// Delete -> login fails, list shrinks.
	resp, _ = doJSON(t, http.MethodDelete, env.ts.URL+"/users/alice", adminTok, nil)
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("delete = %d", resp.StatusCode)
	}
	if code, _ := loginJSON(t, env.ts, "alice", "pw2"); code != http.StatusUnauthorized {
		t.Errorf("deleted user login = %d, want 401", code)
	}

	// Unknown target -> 404.
	resp, _ = doJSON(t, http.MethodDelete, env.ts.URL+"/users/ghost", adminTok, nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("delete unknown = %d, want 404", resp.StatusCode)
	}

	// Persisted to AUTH.yaml.
	b, err := os.ReadFile(env.authPath)
	if err != nil {
		t.Fatalf("read auth file: %v", err)
	}
	if !bytes.Contains(b, []byte("boss")) || bytes.Contains(b, []byte("alice")) {
		t.Errorf("AUTH.yaml persistence unexpected:\n%s", b)
	}
}

func TestLastAdminViaAPI(t *testing.T) {
	env := newUsersTestEnv(t, "boss:\n  password: b\n  role: admin\n")
	_, login := loginJSON(t, env.ts, "boss", "b")
	tok := login["token"].(string)

	resp, _ := doJSON(t, http.MethodDelete, env.ts.URL+"/users/boss", tok, nil)
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("delete last admin = %d, want 409", resp.StatusCode)
	}
	resp, _ = doJSON(t, http.MethodPut, env.ts.URL+"/users/boss", tok, map[string]any{"role": "user"})
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("demote last admin = %d, want 409", resp.StatusCode)
	}
	// Still able to log in.
	if code, _ := loginJSON(t, env.ts, "boss", "b"); code != http.StatusOK {
		t.Errorf("admin login after failed ops = %d", code)
	}
}

// TestHotReloadExternalEdit covers the end-to-end hot update path: an external
// AUTH.yaml edit is picked up by the running store (the same check the
// background watcher performs) and immediately affects logins over HTTP.
func TestHotReloadExternalEdit(t *testing.T) {
	env := newUsersTestEnv(t, "boss:\n  password: b\n  role: admin\n")

	if code, _ := loginJSON(t, env.ts, "newcomer", "n"); code != http.StatusUnauthorized {
		t.Fatalf("newcomer should not exist yet: %d", code)
	}
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(env.authPath, []byte("boss:\n  password: b2\n  role: admin\nnewcomer: n\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}

	changed, err := env.srv.auth.ReloadIfChanged()
	if err != nil || !changed {
		t.Fatalf("ReloadIfChanged = %v, %v", changed, err)
	}

	if code, _ := loginJSON(t, env.ts, "newcomer", "n"); code != http.StatusOK {
		t.Errorf("newcomer login after reload = %d, want 200", code)
	}
	if code, _ := loginJSON(t, env.ts, "boss", "b"); code != http.StatusUnauthorized {
		t.Errorf("old boss password after reload = %d, want 401", code)
	}
	if code, _ := loginJSON(t, env.ts, "boss", "b2"); code != http.StatusOK {
		t.Errorf("new boss password after reload = %d, want 200", code)
	}
}
