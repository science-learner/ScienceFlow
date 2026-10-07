package auth

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// User is a single account with an optional source whitelist.
// Sources is nil or contains "*" when the user may read every source.
type User struct {
	Password string
	Sources  []string
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

// Store holds user accounts, their ACLs and the token signing secret.
type Store struct {
	users  map[string]*User
	secret []byte
}

// Load parses AUTH.yaml. Two shapes are accepted per entry:
//
//	admin: admin123
//	viewer:
//	  password: viewer456
//	  sources: ["app-logs", "plain-text"]
func Load(path string, secret []byte) (*Store, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

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
	if len(users) == 0 {
		return nil, fmt.Errorf("no users found in %s", path)
	}

	if len(secret) == 0 {
		secret = make([]byte, 32)
		if _, err := rand.Read(secret); err != nil {
			return nil, fmt.Errorf("generate secret: %w", err)
		}
	}

	return &Store{users: users, secret: secret}, nil
}

func parseUser(v interface{}) (*User, error) {
	u := &User{}
	switch t := v.(type) {
	case string:
		u.Password = t
	case map[string]interface{}:
		if p, ok := t["password"]; ok {
			ps, ok := p.(string)
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
	default:
		return nil, fmt.Errorf("value must be a password string or a mapping")
	}
	if u.Password == "" {
		return nil, fmt.Errorf("missing password")
	}
	return u, nil
}

func (s *Store) Verify(user, pass string) bool {
	if s == nil {
		return true
	}
	u, ok := s.users[user]
	if !ok {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(u.Password), []byte(pass)) == 1
}

func (s *Store) User(name string) (*User, bool) {
	if s == nil {
		return nil, false
	}
	u, ok := s.users[name]
	return u, ok
}

func (s *Store) Count() int {
	if s == nil {
		return 0
	}
	return len(s.users)
}

// AllowedSources returns the explicit source whitelist for a user, or nil
// when the user may read every source.
func (s *Store) AllowedSources(user string) []string {
	u, ok := s.users[user]
	if !ok || u.allowsAll() {
		return nil
	}
	return append([]string(nil), u.Sources...)
}

// CanAccess reports whether the user may read the given source name.
func (s *Store) CanAccess(user, source string) bool {
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

// IssueToken returns an HMAC-signed token encoding user + expiry.
func (s *Store) IssueToken(user string, ttl time.Duration) (string, time.Time) {
	exp := time.Now().Add(ttl)
	payload := user + "\x00" + strconv.FormatInt(exp.Unix(), 10)

	mac := hmac.New(sha256.New, s.secret)
	mac.Write([]byte(payload))
	sig := mac.Sum(nil)

	tok := base64.RawURLEncoding.EncodeToString([]byte(payload)) + "." +
		base64.RawURLEncoding.EncodeToString(sig)
	return tok, exp
}

// VerifyToken validates signature and expiry, returning the user name.
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
	i := strings.LastIndex(str, "\x00")
	if i < 0 {
		return "", false
	}
	user := str[:i]
	exp, err := strconv.ParseInt(str[i+1:], 10, 64)
	if err != nil || time.Now().Unix() > exp {
		return "", false
	}
	if _, ok := s.users[user]; !ok {
		return "", false
	}
	return user, true
}
