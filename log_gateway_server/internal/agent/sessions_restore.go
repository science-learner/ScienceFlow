// Session discovery for gateway restarts: scans the workspace root for
// per-user task directories and reconstructs the metadata the gateway can
// derive from disk (session id, owner, display name, last activity). The
// server layer turns these into restored in-memory sessions.

package agent

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// HistoricalSession is one session directory found under the workspace root.
type HistoricalSession struct {
	ID         string
	User       string
	Name       string
	LastActive time.Time
}

// DiscoverSessions scans <root>/<user>/<session> directories written by the
// gateway and returns one entry per session that still has task content on
// disk (first-user file, a gateway manifest, RAW.log or the interaction
// transcript). The special task_logs directory is skipped, as are empty
// shells without any task artifact. Entries are newest first.
func DiscoverSessions(root string) ([]HistoricalSession, error) {
	userEntries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("read workspace root: %w", err)
	}

	var out []HistoricalSession
	for _, ue := range userEntries {
		if !ue.IsDir() || ue.Name() == "task_logs" {
			continue
		}
		user := ue.Name()
		sessionEntries, err := os.ReadDir(filepath.Join(root, user))
		if err != nil {
			continue
		}
		for _, se := range sessionEntries {
			if !se.IsDir() {
				continue
			}
			id := se.Name()
			name, last := sessionMetaFromDisk(root, user, id)
			if last.IsZero() {
				continue
			}
			out = append(out, HistoricalSession{
				ID:         id,
				User:       user,
				Name:       name,
				LastActive: last,
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LastActive.After(out[j].LastActive) })
	return out, nil
}

// sessionMetaFromDisk derives (name, lastActive) for one session directory by
// looking at the artifacts the gateway writes: gateway_first_user.txt (lite
// query), the lite/heavy manifests (task text), RAW.log and the interaction
// transcript (timestamps).
func sessionMetaFromDisk(root, user, id string) (string, time.Time) {
	sessDir := filepath.Join(root, user, id)
	var name string
	var last time.Time
	touch := func(path string) {
		if fi, err := os.Stat(path); err == nil && !fi.IsDir() && fi.ModTime().After(last) {
			last = fi.ModTime()
		}
	}

	firstUser := filepath.Join(sessDir, FirstUserQueryFile)
	if b, err := os.ReadFile(firstUser); err == nil {
		name = sessionNameFromQuery(string(b))
		touch(firstUser)
	}
	for _, manifestName := range []string{LiteManifestName, HeavyManifestName} {
		p := filepath.Join(sessDir, manifestName)
		if b, err := os.ReadFile(p); err == nil {
			if name == "" {
				name = sessionNameFromManifest(b)
			}
			touch(p)
		}
	}
	touch(filepath.Join(root, "task_logs", user, id, "RAW.log"))
	touch(filepath.Join(sessDir, RunDirName, ".logs", "interaction.log"))
	return name, last
}

// sessionNameFromManifest extracts tasks[0].task from a gateway manifest.
func sessionNameFromManifest(b []byte) string {
	var doc struct {
		Tasks []struct {
			Task string `yaml:"task"`
		} `yaml:"tasks"`
	}
	if err := yaml.Unmarshal(b, &doc); err != nil || len(doc.Tasks) == 0 {
		return ""
	}
	return sessionNameFromQuery(doc.Tasks[0].Task)
}

// sessionNameFromQuery mirrors session.Manager.MaybeNameFromQuery: first line,
// whitespace-collapsed, capped at 120 runes, so restored and live sessions
// display identically.
func sessionNameFromQuery(query string) string {
	line := strings.TrimSpace(query)
	if i := strings.IndexAny(line, "\r\n"); i >= 0 {
		line = line[:i]
	}
	line = strings.Join(strings.Fields(line), " ")
	if runes := []rune(line); len(runes) > 120 {
		line = string(runes[:120])
	}
	return line
}
