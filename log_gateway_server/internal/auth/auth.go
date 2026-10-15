// Package auth implements the gateway's account store: AUTH.yaml parsing,
// password verification, HMAC tokens, source ACLs, and the admin-facing user
// management API backing store.
//
// The store is safe for concurrent use. Accounts can be changed in two ways,
// both effective without a restart (hot reload):
//
//   - editing AUTH.yaml directly: a background watcher polls the file's
//     mtime/size and swaps the in-memory user map when it changes;
//   - the /users admin API (see internal/server/users.go), which mutates the
//     store and atomically rewrites AUTH.yaml.
//
// Tokens bind the user name and a password-derived stamp, so changing or
// deleting an account immediately invalidates that account's live tokens.
package auth

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// Role values. RoleAdmin users may call the /users management API.
const (
	RoleUser  = "user"
	RoleAdmin = "admin"
)

// User is a single account with an optional source whitelist.
// Sources is nil or contains "*" when the user may read every source.
type User struct {
	Password string
	Sources  []string
	Role     string // "" or RoleUser = normal user; RoleAdmin = super admin
}

func (u *User) allowsAll() bool {
	if len(u.Sources) == 0 {
		return true
	}
	for _, s := range u.Sources {
		if s == "*" {
			return true
		}
	}
	return false
}

// IsAdmin reports whether the account may manage other accounts.
func (u *User) IsAdmin() bool { return u.Role == RoleAdmin }

func (u *User) displayRole() string {
	if u.Role == RoleAdmin {
		return RoleAdmin
	}
	return RoleUser
}

// UserView is the API projection of an account (never carries the password).
type UserView struct {
	Name    string   `json:"name"`
	Role    string   `json:"role"`
	Admin   bool     `json:"admin"`
	Sources []string `json:"sources"` // nil = all sources
}

// Sentinel errors surfaced by account management; the HTTP layer maps them
// to proper status codes.
var (
	ErrUserNotFound = errors.New("用户不存在")
	ErrUserExists   = errors.New("用户已存在")
	ErrLastAdmin    = errors.New("不能删除或降级最后一个管理员账号")
)

// Store holds user accounts, their ACLs and the token signing secret.
type Store struct {
	mu     sync.RWMutex
	path   string
	users  map[string]*User
	secret []byte

	// File identity captured at load/save, used by ReloadIfChanged to detect
	// external edits without re-reading the file on every poll.
	modTime time.Time
	size    int64
}

// Load parses AUTH.yaml. Two shapes are accepted per entry:
//
//	admin: admin123
//	viewer:
//	  password: viewer456
//	  role: admin            # optional: super admin (account management API)
//	  sources: ["app-logs", "plain-text"]
func Load(path string, secret []byte) (*Store, error) {
	users, modTime, size, err := readUsersFile(path)
	if err != nil {
		return nil, err
	}
	if len(users) == 0 {
		return nil, fmt.Errorf("no users found in %s", path)
	}

	if len(secret) == 0 {
		secret = make([]byte, 32)
		if _, err := rand.Read(secret); err != nil {
			return nil, fmt.Errorf("generate secret: %w", err)
		}
	}

	return &Store{
		path:    path,
		users:   users,
		secret:  secret,
		modTime: modTime,
		size:    size,
	}, nil
}

// readUsersFile parses path and returns the users plus the file's mtime/size.
func readUsersFile(path string) (map[string]*User, time.Time, int64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, time.Time{}, 0, err
	}
	users, err := parseUsers(b, path)
	if err != nil {
		return nil, time.Time{}, 0, err
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, time.Time{}, 0, err
	}
	return users, fi.ModTime(), fi.Size(), nil
}

// parseUsers decodes the YAML body into accounts. path is only used in error
// messages.
func parseUsers(b []byte, path string) (map[string]*User, error) {
	var raw map[string]interface{}
	if err := yaml.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	users := make(map[string]*User, len(raw))
	for name, v := range raw {
		u, err := parseUser(v)
		if err != nil {
			return nil, fmt.Errorf("%s: user %q: %w", path, name, err)
		}
		users[name] = u
	}
	return users, nil
}

// yamlScalarString coerces a YAML scalar to its string form. Hand-edited
// files frequently contain unquoted numeric/boolean passwords
// (`password: 123456`), which yaml.v3 decodes as int/bool — without this
// coercion the whole AUTH.yaml would fail to load.
func yamlScalarString(v interface{}) (string, bool) {
	switch t := v.(type) {
	case string:
		return t, true
	case int:
		return strconv.Itoa(t), true
	case int64:
		return strconv.FormatInt(t, 10), true
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64), true
	case bool:
		return strconv.FormatBool(t), true
	default:
		return "", false
	}
}

func parseUser(v interface{}) (*User, error) {
	u := &User{}
	switch t := v.(type) {
	case map[string]interface{}:
		if p, ok := t["password"]; ok {
			ps, ok := yamlScalarString(p)
			if !ok {
				return nil, fmt.Errorf("password must be a string")
			}
			u.Password = ps
		}
		if src, ok := t["sources"]; ok {
			switch s := src.(type) {
			case []interface{}:
				for _, item := range s {
					if str, ok := item.(string); ok {
						u.Sources = append(u.Sources, str)
					}
				}
			case []string:
				u.Sources = s
			}
		}
		if r, ok := t["role"]; ok {
			rs, ok := r.(string)
			if !ok {
				return nil, fmt.Errorf("role must be a string")
			}
			role, err := normalizeRole(rs)
			if err != nil {
				return nil, err
			}
			u.Role = role
		}
	default:
		ps, ok := yamlScalarString(v)
		if !ok {
			return nil, fmt.Errorf("value must be a password string or a mapping")
		}
		u.Password = ps
	}
	if u.Password == "" {
		return nil, fmt.Errorf("missing password")
	}
	return u, nil
}

// normalizeRole maps the YAML/JSON role spelling to a canonical value.
func normalizeRole(role string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "", "user":
		return RoleUser, nil
	case "admin", "superadmin", "super_admin":
		return RoleAdmin, nil
	default:
		return "", fmt.Errorf("role must be %q or %q", RoleAdmin, RoleUser)
	}
}

// ---------------------------------------------------------------------------
// Hot reload
// ---------------------------------------------------------------------------

// ReloadIfChanged re-reads AUTH.yaml when its mtime or size differs from the
// last known state. Invalid content is rejected: the error is returned and the
// current accounts stay active. Existing tokens keep working because the
// signing secret is never touched by a reload.
func (s *Store) ReloadIfChanged() (bool, error) {
	fi, err := os.Stat(s.path)
	if err != nil {
		return false, err
	}
	s.mu.RLock()
	unchanged := fi.ModTime().Equal(s.modTime) && fi.Size() == s.size
	s.mu.RUnlock()
	if unchanged {
		return false, nil
	}

	users, modTime, size, err := readUsersFile(s.path)
	if err != nil {
		return false, err
	}
	if len(users) == 0 {
		return false, fmt.Errorf("%s: no users found (ignoring reload)", s.path)
	}

	s.mu.Lock()
	s.users = users
	s.modTime = modTime
	s.size = size
	s.mu.Unlock()
	return true, nil
}

// WatchReload polls AUTH.yaml every interval and applies external edits.
// The returned stop function terminates the watcher. Reload failures (invalid
// YAML) are logged once per distinct error and never crash the gateway.
func (s *Store) WatchReload(interval time.Duration, logger *log.Logger) (stop func()) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	done := make(chan struct{})
	var once sync.Once
	var lastErr string

	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-done:
				return
			case <-t.C:
				changed, err := s.ReloadIfChanged()
				if err != nil {
					if msg := err.Error(); logger != nil && msg != lastErr {
						lastErr = msg
						logger.Printf("auth reload %s: %v (keeping current accounts)", s.path, msg)
					}
					continue
				}
				lastErr = ""
				if changed && logger != nil {
					logger.Printf("auth reloaded: %d user(s) from %s", s.Count(), s.path)
				}
			}
		}
	}()
	return func() { once.Do(func() { close(done) }) }
}

// ---------------------------------------------------------------------------
// Reads (concurrency-safe)
// ---------------------------------------------------------------------------

func (s *Store) Verify(user, pass string) bool {
	if s == nil {
		return true
	}
	s.mu.RLock()
	u, ok := s.users[user]
	s.mu.RUnlock()
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(u.Password), []byte(pass)) == 1
}

func (s *Store) User(name string) (*User, bool) {
	if s == nil {
		return nil, false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.users[name]
	return u, ok
}

func (s *Store) Count() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.users)
}

// IsAdmin reports whether the account exists and carries the admin role.
func (s *Store) IsAdmin(name string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.users[name]
	return ok && u.IsAdmin()
}

// AllowedSources returns the explicit source whitelist for a user, or nil
// when the user may read every source.
func (s *Store) AllowedSources(user string) []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.users[user]
	if !ok || u.allowsAll() {
		return nil
	}
	return append([]string(nil), u.Sources...)
}

// CanAccess reports whether the user may read the given source name.
func (s *Store) CanAccess(user, source string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	u, ok := s.users[user]
	if !ok {
		return false
	}
	if u.allowsAll() {
		return true
	}
	for _, v := range u.Sources {
		if v == source {
			return true
		}
	}
	return false
}

// ListUsers returns all accounts (without passwords), sorted by name.
func (s *Store) ListUsers() []UserView {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]UserView, 0, len(s.users))
	for name, u := range s.users {
		var sources []string
		if len(u.Sources) > 0 {
			sources = append([]string(nil), u.Sources...)
		}
		out = append(out, UserView{
			Name:    name,
			Role:    u.displayRole(),
			Admin:   u.IsAdmin(),
			Sources: sources,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ---------------------------------------------------------------------------
// Account management (persists AUTH.yaml atomically)
// ---------------------------------------------------------------------------

// UserPatch carries partial changes for UpdateUser; nil fields stay unchanged.
type UserPatch struct {
	Password *string
	Sources  *[]string
	Role     *string
}

var usernameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

// ValidUsername reports whether name is acceptable for a new account. It must
// be filesystem-safe because user names become workspace directory names.
func ValidUsername(name string) bool { return usernameRe.MatchString(name) }

func validatePassword(pw string) error {
	if pw == "" {
		return errors.New("密码不能为空")
	}
	if len(pw) > 256 {
		return errors.New("密码过长（最多 256 字节）")
	}
	if strings.ContainsAny(pw, "\x00\n\r") {
		return errors.New("密码不能包含换行或空字符")
	}
	return nil
}

// cleanSources copies and trims an ACL list, treating "*" as the universal
// marker and dropping empty entries.
func cleanSources(in []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

// CreateUser adds a new account and persists the file.
func (s *Store) CreateUser(name, password string, sources []string, role string) error {
	if s == nil {
		return errors.New("auth is not enabled")
	}
	if !ValidUsername(name) {
		return errors.New("用户名需以字母或数字开头，仅可包含字母、数字、点、下划线和连字符（最长 64 字符）")
	}
	if err := validatePassword(password); err != nil {
		return err
	}
	canon, err := normalizeRole(role)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.users[name]; exists {
		return ErrUserExists
	}
	s.users[name] = &User{Password: password, Sources: cleanSources(sources), Role: canon}
	if err := s.saveLocked(); err != nil {
		delete(s.users, name)
		return err
	}
	return nil
}

// UpdateUser applies a partial change and persists the file. The last
// remaining admin cannot be demoted (lockout prevention).
func (s *Store) UpdateUser(name string, patch UserPatch) error {
	if s == nil {
		return errors.New("auth is not enabled")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.users[name]
	if !ok {
		return ErrUserNotFound
	}

	next := *u
	if patch.Password != nil {
		if err := validatePassword(*patch.Password); err != nil {
			return err
		}
		next.Password = *patch.Password
	}
	if patch.Sources != nil {
		next.Sources = cleanSources(*patch.Sources)
	}
	if patch.Role != nil {
		canon, err := normalizeRole(*patch.Role)
		if err != nil {
			return err
		}
		if u.IsAdmin() && canon != RoleAdmin && s.adminCountLocked() <= 1 {
			return ErrLastAdmin
		}
		next.Role = canon
	}

	s.users[name] = &next
	if err := s.saveLocked(); err != nil {
		s.users[name] = u
		return err
	}
	return nil
}

// DeleteUser removes an account and persists the file. The last remaining
// admin cannot be deleted (lockout prevention).
func (s *Store) DeleteUser(name string) error {
	if s == nil {
		return errors.New("auth is not enabled")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	u, ok := s.users[name]
	if !ok {
		return ErrUserNotFound
	}
	if u.IsAdmin() && s.adminCountLocked() <= 1 {
		return ErrLastAdmin
	}

	delete(s.users, name)
	if err := s.saveLocked(); err != nil {
		s.users[name] = u
		return err
	}
	return nil
}

func (s *Store) adminCountLocked() int {
	n := 0
	for _, u := range s.users {
		if u.IsAdmin() {
			n++
		}
	}
	return n
}

// authFileHeader is written above the marshaled accounts on every save so the
// file stays self-documenting even though YAML comments cannot be preserved.
const authFileHeader = "# 用户鉴权配置（由 lgw 启动引导/用户管理接口维护，热更新无需重启）\n" +
	"# 写法：name: password  或  name: {password: xxx, role: admin, sources: [\"src\", \"*\"]}\n" +
	"# role: admin 为超级管理员，可调用 /users 接口增删改账号。\n"

// saveLocked writes the current accounts to AUTH.yaml atomically (temp file +
// rename) with owner-only permissions. Callers hold s.mu.
func (s *Store) saveLocked() error {
	out := make(map[string]interface{}, len(s.users))
	for name, u := range s.users {
		if u.Role != RoleAdmin && len(u.Sources) == 0 {
			out[name] = u.Password
			continue
		}
		m := map[string]interface{}{"password": u.Password}
		if u.Role == RoleAdmin {
			m["role"] = RoleAdmin
		}
		if len(u.Sources) > 0 {
			m["sources"] = u.Sources
		}
		out[name] = m
	}
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(out); err != nil {
		return fmt.Errorf("marshal %s: %w", s.path, err)
	}
	_ = enc.Close()

	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, append([]byte(authFileHeader), buf.Bytes()...), 0o600); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("replace %s: %w", s.path, err)
	}
	// Refresh the file identity so the watcher does not re-apply our own write.
	if fi, err := os.Stat(s.path); err == nil {
		s.modTime = fi.ModTime()
		s.size = fi.Size()
	}
	return nil
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

// stampOf derives a short token stamp from the account password so that
// password changes invalidate live tokens. Returns "" for unknown users.
func (s *Store) stampOf(name string) string {
	u, ok := s.users[name]
	if !ok {
		return ""
	}
	sum := sha256.Sum256([]byte(name + "\x00" + u.Password))
	return hex.EncodeToString(sum[:8])
}

// IssueToken returns an HMAC-signed token encoding user + expiry + a password
// stamp (password changes revoke all outstanding tokens).
func (s *Store) IssueToken(user string, ttl time.Duration) (string, time.Time) {
	exp := time.Now().Add(ttl)
	s.mu.RLock()
	stamp := s.stampOf(user)
	s.mu.RUnlock()
	payload := user + "\x00" + strconv.FormatInt(exp.Unix(), 10) + "\x00" + stamp

	mac := hmac.New(sha256.New, s.secret)
	mac.Write([]byte(payload))
	sig := mac.Sum(nil)

	tok := base64.RawURLEncoding.EncodeToString([]byte(payload)) + "." +
		base64.RawURLEncoding.EncodeToString(sig)
	return tok, exp
}

// VerifyToken validates signature, expiry, account existence and the password
// stamp, returning the user name.
func (s *Store) VerifyToken(tok string) (string, bool) {
	if s == nil {
		return "", false
	}
	parts := strings.SplitN(tok, ".", 2)
	if len(parts) != 2 {
		return "", false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return "", false
	}
	sig, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", false
	}

	mac := hmac.New(sha256.New, s.secret)
	mac.Write(payload)
	expect := mac.Sum(nil)
	if !hmac.Equal(expect, sig) {
		return "", false
	}

	str := string(payload)
	fields := strings.Split(str, "\x00")
	if len(fields) != 3 {
		return "", false
	}
	user := fields[0]
	exp, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || time.Now().Unix() > exp {
		return "", false
	}

	s.mu.RLock()
	stamp := s.stampOf(user)
	s.mu.RUnlock()
	if subtle.ConstantTimeCompare([]byte(stamp), []byte(fields[2])) != 1 {
		return "", false
	}
	return user, true
}
