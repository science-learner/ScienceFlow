package auth

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func writeAuthFile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "AUTH.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}
	return path
}

func TestLoadAndVerify(t *testing.T) {
	path := writeAuthFile(t, "admin: admin123\nviewer:\n  password: viewer456\n  sources: [\"prefixed-logs\"]\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !s.Verify("admin", "admin123") {
		t.Error("admin/admin123 should verify")
	}
	if s.Verify("admin", "wrong") {
		t.Error("wrong password must fail")
	}
	if s.Verify("ghost", "x") {
		t.Error("unknown user must fail")
	}
	if s.Count() != 2 {
		t.Errorf("Count = %d, want 2", s.Count())
	}
}

func TestNumericScalarPasswords(t *testing.T) {
	// Hand-edited YAML often leaves numeric passwords unquoted; yaml.v3
	// decodes them as int/bool and they must still load.
	path := writeAuthFile(t, "ictmind:\n  password: 123456\n  role: admin\nplain: 654321\nflag: true\n")
	s, err := Load(path, nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !s.Verify("ictmind", "123456") {
		t.Error("unquoted numeric mapping password should verify as string")
	}
	if !s.Verify("plain", "654321") {
		t.Error("unquoted numeric scalar password should verify as string")
	}
	if !s.Verify("flag", "true") {
		t.Error("unquoted boolean password should verify as string")
	}
	if !s.IsAdmin("ictmind") {
		t.Error("ictmind should be admin")
	}
}

func TestRoleParsing(t *testing.T) {
	path := writeAuthFile(t, "boss:\n  password: x\n  role: admin\nplain: p\nboo:\n  password: q\n  role: user\n")
	s, err := Load(path, nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !s.IsAdmin("boss") {
		t.Error("boss should be admin")
	}
	if s.IsAdmin("plain") || s.IsAdmin("boo") {
		t.Error("plain/boo must not be admin")
	}
	// Invalid role spelling is rejected at parse time.
	bad := writeAuthFile(t, "a:\n  password: x\n  role: superuser\n")
	if _, err := Load(bad, nil); err == nil {
		t.Error("unknown role must fail to parse")
	}
}

func TestTokenInvalidatedByPasswordChange(t *testing.T) {
	path := writeAuthFile(t, "admin: admin123\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok, _ := s.IssueToken("admin", time.Hour)
	if user, ok := s.VerifyToken(tok); !ok || user != "admin" {
		t.Fatalf("token should verify, got %q %v", user, ok)
	}

	pw := "newpass"
	if err := s.UpdateUser("admin", UserPatch{Password: &pw}); err != nil {
		t.Fatalf("UpdateUser: %v", err)
	}
	if _, ok := s.VerifyToken(tok); ok {
		t.Error("token must be invalidated after password change")
	}
	if !s.Verify("admin", "newpass") {
		t.Error("new password should verify")
	}
}

func TestTokenInvalidatedByDelete(t *testing.T) {
	path := writeAuthFile(t, "admin:\n  password: x\n  role: admin\nvictim: v\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok, _ := s.IssueToken("victim", time.Hour)
	if _, ok := s.VerifyToken(tok); !ok {
		t.Fatal("token should verify before delete")
	}
	if err := s.DeleteUser("victim"); err != nil {
		t.Fatalf("DeleteUser: %v", err)
	}
	if _, ok := s.VerifyToken(tok); ok {
		t.Error("token must be invalidated after account deletion")
	}
}

func TestReloadIfChanged(t *testing.T) {
	path := writeAuthFile(t, "admin: admin123\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok, _ := s.IssueToken("admin", time.Hour)

	// No change -> false.
	if changed, err := s.ReloadIfChanged(); err != nil || changed {
		t.Fatalf("unchanged reload = %v, %v", changed, err)
	}

	// External edit: change password and add a user. Ensure a different mtime.
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(path, []byte("admin: changed456\nnewbie:\n  password: n\n  role: admin\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	changed, err := s.ReloadIfChanged()
	if err != nil || !changed {
		t.Fatalf("reload = %v, %v; want changed", changed, err)
	}
	if !s.Verify("admin", "changed456") || s.Verify("admin", "admin123") {
		t.Error("reloaded password not applied")
	}
	if !s.Verify("newbie", "n") || !s.IsAdmin("newbie") {
		t.Error("reloaded user not applied")
	}
	if _, ok := s.VerifyToken(tok); ok {
		t.Error("token must be invalidated by reloaded password change")
	}

	// Invalid YAML: error returned, old accounts kept.
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(path, []byte("admin: [not-a-string\n"), 0o600); err != nil {
		t.Fatalf("rewrite invalid: %v", err)
	}
	if _, err := s.ReloadIfChanged(); err == nil {
		t.Error("invalid YAML should return an error")
	}
	if !s.Verify("admin", "changed456") {
		t.Error("accounts must survive an invalid reload")
	}
}

func TestWatchReload(t *testing.T) {
	path := writeAuthFile(t, "admin: a\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	stop := s.WatchReload(20*time.Millisecond, nil)
	defer stop()

	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(path, []byte("admin: a\nwatcher_user: w\n"), 0o600); err != nil {
		t.Fatalf("rewrite: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s.Verify("watcher_user", "w") {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("watcher did not pick up the external edit")
}

func TestCreateUserPersistsAndValidates(t *testing.T) {
	path := writeAuthFile(t, "admin:\n  password: a\n  role: admin\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if err := s.CreateUser("worker", "pw", []string{"prefixed-logs"}, "user"); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	if !s.Verify("worker", "pw") {
		t.Error("new user should verify")
	}
	if err := s.CreateUser("worker", "x", nil, "user"); err == nil {
		t.Error("duplicate user must fail")
	}
	if err := s.CreateUser("bad name", "x", nil, "user"); err == nil {
		t.Error("invalid username must fail")
	}
	if err := s.CreateUser("empty", "", nil, "user"); err == nil {
		t.Error("empty password must fail")
	}
	if err := s.CreateUser("weird", "x", nil, "root"); err == nil {
		t.Error("invalid role must fail")
	}

	// Reload from disk with a fresh store: the new account must persist.
	s2, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("reload store: %v", err)
	}
	if !s2.Verify("worker", "pw") {
		t.Error("created user must be persisted to AUTH.yaml")
	}
	if s2.CanAccess("worker", "other") {
		t.Error("worker ACL should be restricted to prefixed-logs")
	}
}

func TestLastAdminProtection(t *testing.T) {
	path := writeAuthFile(t, "admin:\n  password: a\n  role: admin\nuser1: u\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := s.DeleteUser("admin"); err == nil {
		t.Error("deleting the last admin must fail")
	}
	role := "user"
	if err := s.UpdateUser("admin", UserPatch{Role: &role}); err == nil {
		t.Error("demoting the last admin must fail")
	}

	// With a second admin, both operations succeed.
	if err := s.CreateUser("admin2", "b", nil, "admin"); err != nil {
		t.Fatalf("CreateUser admin2: %v", err)
	}
	if err := s.DeleteUser("admin"); err != nil {
		t.Errorf("deleting a non-last admin should succeed: %v", err)
	}
	if err := s.UpdateUser("admin2", UserPatch{Role: &role}); err == nil {
		t.Error("demoting the last remaining admin must still fail")
	}
}

func TestListUsers(t *testing.T) {
	path := writeAuthFile(t, "z: z\nadmin:\n  password: a\n  role: admin\n")
	s, err := Load(path, nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	views := s.ListUsers()
	if len(views) != 2 || views[0].Name != "admin" || views[1].Name != "z" {
		t.Fatalf("ListUsers = %+v, want sorted admin,z", views)
	}
	if !views[0].Admin || views[0].Role != "admin" {
		t.Errorf("admin view = %+v", views[0])
	}
	if views[1].Admin || views[1].Role != "user" {
		t.Errorf("normal user view = %+v", views[1])
	}
}

// TestSaveFormat checks the human-facing file shape: password-only accounts
// stay scalar, admin/ACL accounts become mappings with a header comment.
func TestSaveFormat(t *testing.T) {
	path := writeAuthFile(t, "admin:\n  password: a\n  role: admin\n")
	s, err := Load(path, []byte("secret"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := s.CreateUser("plain", "p", nil, "user"); err != nil {
		t.Fatalf("CreateUser: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	text := string(b)
	if !strings.HasPrefix(text, "# 用户鉴权配置") {
		t.Error("file should start with the header comment")
	}
	var raw map[string]interface{}
	if err := yaml.Unmarshal(b, &raw); err != nil {
		t.Fatalf("saved file is not valid YAML: %v", err)
	}
	if pw, ok := raw["plain"].(string); !ok || pw != "p" {
		t.Errorf("password-only account should marshal as scalar, got %#v", raw["plain"])
	}
	m, ok := raw["admin"].(map[string]interface{})
	if !ok || m["role"] != "admin" || m["password"] != "a" {
		t.Errorf("admin mapping = %#v", raw["admin"])
	}
}

func TestUpdatedSourcesReplaceACL(t *testing.T) {
	path := writeAuthFile(t, "u:\n  password: p\n  sources: [\"a\"]\n")
	s, err := Load(path, nil)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	srcs := []string{"b", "c", " b "}
	if err := s.UpdateUser("u", UserPatch{Sources: &srcs}); err != nil {
		t.Fatalf("UpdateUser: %v", err)
	}
	got := s.AllowedSources("u")
	if len(got) != 2 || got[0] != "b" || got[1] != "c" {
		t.Errorf("AllowedSources = %v, want [b c]", got)
	}
	if s.CanAccess("u", "a") || !s.CanAccess("u", "b") {
		t.Error("ACL not replaced")
	}
}
