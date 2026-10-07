package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// The gateway manages the SAME model registry file the installed scienceflow
// CLI reads: ~/.config/scienceflow/models.json (or
// $XDG_CONFIG_HOME/scienceflow/models.json). Schema v1:
//
//	{
//	  "version": 1,
//	  "models": {
//	    "<alias>": {
//	      "model": "<model name>",
//	      "reasoning_replay": "preserve",
//	      "endpoints": { "code": [{"url","key"}], "feedback": [{"url","key"}] }
//	    }
//	  },
//	  "defaults": { "code_models": ["<alias>"], "feedback_models": ["<alias>"], "selection": "auto" }
//	}
//
// The REST-facing types (ModelView / Create / Update requests) are unchanged;
// the mapping is: ID <-> alias, ModelName <-> model, APIURL/APIKey <-> primary
// (code[0]) endpoint, Active <-> head of defaults.code_models. Per-session
// stage overrides are resolved at task launch by materializing a per-task
// registry copy (see WriteSessionRegistry / Runner.buildEnv).
//
// Invariant: whenever models exist, defaults.code_models / feedback_models
// must be non-empty — an existing registry with empty defaults makes the CLI
// fail its config load ("defaults.<role>_models must select at least one
// alias").

type registryEndpoint struct {
	URL string `json:"url"`
	Key string `json:"key"`
}

// registryModel keeps unknown fields (pricing, extra endpoints) verbatim via
// RawMessage so gateway edits never clobber hand-curated registry entries.
type registryModel struct {
	Model           string          `json:"model"`
	ReasoningReplay string          `json:"reasoning_replay,omitempty"`
	Pricing         json.RawMessage `json:"pricing,omitempty"`
	Endpoints       json.RawMessage `json:"endpoints,omitempty"`
}

type registryDefaults struct {
	CodeModels     []string `json:"code_models,omitempty"`
	FeedbackModels []string `json:"feedback_models,omitempty"`
	Selection      string   `json:"selection,omitempty"`
}

type registryFile struct {
	Version  int                      `json:"version"`
	Models   map[string]registryModel `json:"models"`
	Defaults registryDefaults         `json:"defaults"`
}

type ModelView struct {
	ID           string `json:"id"`
	ModelName    string `json:"model_name"`
	APIKey       string `json:"api_key"`
	APIKeyMasked string `json:"api_key_masked"`
	APIURL       string `json:"api_url"`
	Active       bool   `json:"is_active"`
	CreatedAt    string `json:"created_at,omitempty"`
	UpdatedAt    string `json:"updated_at,omitempty"`
}

type ModelCreateRequest struct {
	ModelName string `json:"model_name"`
	APIKey    string `json:"api_key"`
	APIURL    string `json:"api_url"`
}

type ModelUpdateRequest struct {
	ModelName string `json:"model_name"`
	APIKey    string `json:"api_key"`
	APIURL    string `json:"api_url"`
}

type ModelStore struct {
	mu               sync.RWMutex
	path             string
	reg              registryFile
	active           string
	sessions         map[string]string // session -> main model alias
	sessionsCoder    map[string]string // session -> code stage override
	sessionsFeedback map[string]string // session -> feedback stage override
}

func NewModelStore(path string) (*ModelStore, error) {
	store := &ModelStore{
		path:             path,
		sessions:         make(map[string]string),
		sessionsCoder:    make(map[string]string),
		sessionsFeedback: make(map[string]string),
	}
	if path == "" {
		return store, nil
	}
	if err := store.load(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *ModelStore) load() error {
	b, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read model store: %w", err)
	}
	var reg registryFile
	if err := json.Unmarshal(b, &reg); err != nil {
		return fmt.Errorf("parse model store: %w", err)
	}
	if reg.Models == nil {
		reg.Models = make(map[string]registryModel)
	}
	s.reg = reg
	s.active = firstAlias(s.reg.Defaults.CodeModels, s.reg.Defaults.FeedbackModels)
	return nil
}

func (s *ModelStore) saveLocked() error {
	if s.path == "" {
		return nil
	}
	if len(s.reg.Models) == 0 {
		// An existing registry with no selectable aliases fails the CLI's
		// config load; with no models managed at all, the safest state is no
		// registry file (the CLI then skips registry application entirely).
		_ = os.Remove(s.path)
		return nil
	}
	return writeRegistry(s.path, s.reg)
}

func writeRegistry(path string, reg registryFile) error {
	if reg.Version == 0 {
		reg.Version = 1
	}
	if reg.Models == nil {
		reg.Models = make(map[string]registryModel)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create model store dir: %w", err)
	}
	b, err := json.MarshalIndent(reg, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (s *ModelStore) List() []ModelView {
	s.mu.RLock()
	defer s.mu.RUnlock()
	aliases := make([]string, 0, len(s.reg.Models))
	for alias := range s.reg.Models {
		aliases = append(aliases, alias)
	}
	sort.Strings(aliases)
	items := make([]ModelView, 0, len(aliases))
	for _, alias := range aliases {
		items = append(items, s.viewLocked(alias))
	}
	return items
}

func (s *ModelStore) Create(req ModelCreateRequest) (ModelView, error) {
	name := strings.TrimSpace(req.ModelName)
	if name == "" || strings.TrimSpace(req.APIKey) == "" || strings.TrimSpace(req.APIURL) == "" {
		return ModelView{}, errors.New("model_name, api_key and api_url are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.reg.Models[name]; exists {
		return ModelView{}, errors.New("model already exists")
	}
	if s.reg.Models == nil {
		s.reg.Models = make(map[string]registryModel)
	}
	endpoint := registryEndpoint{URL: strings.TrimSpace(req.APIURL), Key: strings.TrimSpace(req.APIKey)}
	endpoints, err := json.Marshal(map[string][]registryEndpoint{
		"code":     {endpoint},
		"feedback": {endpoint},
	})
	if err != nil {
		return ModelView{}, err
	}
	s.reg.Models[name] = registryModel{Model: name, ReasoningReplay: "preserve", Endpoints: endpoints}
	if s.active == "" {
		s.activateLocked(name)
	}
	if err := s.saveLocked(); err != nil {
		return ModelView{}, err
	}
	return s.viewLocked(name), nil
}

func (s *ModelStore) Update(id string, req ModelUpdateRequest) (ModelView, error) {
	name := strings.TrimSpace(req.ModelName)
	if name == "" || strings.TrimSpace(req.APIURL) == "" {
		return ModelView{}, errors.New("model_name and api_url are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.reg.Models[id]
	if !ok {
		return ModelView{}, errors.New("model not found")
	}
	if name != id {
		if _, dup := s.reg.Models[name]; dup {
			return ModelView{}, errors.New("model already exists")
		}
	}
	m.Model = name
	m.Endpoints = withPrimaryEndpoint(m.Endpoints, strings.TrimSpace(req.APIURL), strings.TrimSpace(req.APIKey))
	if name != id {
		delete(s.reg.Models, id)
	}
	s.reg.Models[name] = m
	if name != id {
		s.renameLocked(id, name)
	}
	if err := s.saveLocked(); err != nil {
		return ModelView{}, err
	}
	return s.viewLocked(name), nil
}

func (s *ModelStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.reg.Models[id]; !ok {
		return errors.New("model not found")
	}
	delete(s.reg.Models, id)
	s.reg.Defaults.CodeModels = removeFromList(s.reg.Defaults.CodeModels, id)
	s.reg.Defaults.FeedbackModels = removeFromList(s.reg.Defaults.FeedbackModels, id)
	for sessionID, modelID := range s.sessions {
		if modelID == id {
			delete(s.sessions, sessionID)
		}
	}
	for sessionID, modelID := range s.sessionsCoder {
		if modelID == id {
			delete(s.sessionsCoder, sessionID)
		}
	}
	for sessionID, modelID := range s.sessionsFeedback {
		if modelID == id {
			delete(s.sessionsFeedback, sessionID)
		}
	}
	// Re-fill defaults from the remaining pool so the registry stays valid
	// for the CLI; promote a new active when the current one was deleted.
	if len(s.reg.Models) > 0 {
		any := firstAlias(sortedAliases(s.reg.Models))
		if len(s.reg.Defaults.CodeModels) == 0 {
			s.reg.Defaults.CodeModels = []string{any}
		}
		if len(s.reg.Defaults.FeedbackModels) == 0 {
			s.reg.Defaults.FeedbackModels = []string{any}
		}
		if s.active == id || s.active == "" {
			s.active = s.reg.Defaults.CodeModels[0]
		}
	}
	return s.saveLocked()
}

func (s *ModelStore) Activate(id string) (ModelView, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.reg.Models[id]; !ok {
		return ModelView{}, errors.New("model not found")
	}
	s.activateLocked(id)
	if err := s.saveLocked(); err != nil {
		return ModelView{}, err
	}
	return s.viewLocked(id), nil
}

func (s *ModelStore) SetSessionModel(session, id string) error {
	if strings.TrimSpace(session) == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.reg.Models[id]; !ok {
		return errors.New("model not found")
	}
	s.sessions[session] = id
	return nil
}

func (s *ModelStore) SetSessionStages(session, coderID, feedbackID string) error {
	session = strings.TrimSpace(session)
	if session == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, stageID := range []string{coderID, feedbackID} {
		if stageID != "" {
			if _, ok := s.reg.Models[stageID]; !ok {
				return errors.New("model not found")
			}
		}
	}
	if coderID != "" {
		s.sessionsCoder[session] = coderID
	} else {
		delete(s.sessionsCoder, session)
	}
	if feedbackID != "" {
		s.sessionsFeedback[session] = feedbackID
	} else {
		delete(s.sessionsFeedback, session)
	}
	return nil
}

// SessionStages returns the raw per-session stage overrides (may be empty).
func (s *ModelStore) SessionStages(session string) (string, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessionsCoder[session], s.sessionsFeedback[session]
}

// SessionStageAliases resolves the code/feedback model aliases a session's
// agent task should run with: per-stage override, then the session's main
// model, then the globally active model. ok is false when both stages resolve
// to the global default — the shared registry already describes the task and
// no per-task materialization is needed.
func (s *ModelStore) SessionStageAliases(session string) (code, feedback string, ok bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	code = s.stageAliasLocked(session, "code")
	feedback = s.stageAliasLocked(session, "feedback")
	if code == "" && feedback == "" {
		return "", "", false
	}
	if code == s.active && feedback == s.active {
		return code, feedback, false
	}
	return code, feedback, true
}

func (s *ModelStore) stageAliasLocked(session, stage string) string {
	alias := ""
	switch stage {
	case "code":
		alias = s.sessionsCoder[session]
	case "feedback":
		alias = s.sessionsFeedback[session]
	}
	if alias == "" {
		alias = s.sessions[session]
	}
	if alias == "" {
		alias = s.active
	}
	return alias
}

// WriteSessionRegistry materializes a per-task copy of the model registry
// under xdgDir as <xdgDir>/scienceflow/models.json with session-specific
// code/feedback defaults. Pointing the subprocess at it via XDG_CONFIG_HOME
// makes the CLI resolve exactly those stage models without touching the
// shared registry.
func (s *ModelStore) WriteSessionRegistry(xdgDir, code, feedback string) error {
	s.mu.RLock()
	reg := registryFile{
		Version: s.reg.Version,
		Models:  make(map[string]registryModel, len(s.reg.Models)),
		Defaults: registryDefaults{
			CodeModels:     append([]string(nil), s.reg.Defaults.CodeModels...),
			FeedbackModels: append([]string(nil), s.reg.Defaults.FeedbackModels...),
			Selection:      s.reg.Defaults.Selection,
		},
	}
	for alias, m := range s.reg.Models {
		reg.Models[alias] = m
	}
	s.mu.RUnlock()

	if code != "" {
		reg.Defaults.CodeModels = moveToFront(reg.Defaults.CodeModels, code)
	}
	if feedback != "" {
		reg.Defaults.FeedbackModels = moveToFront(reg.Defaults.FeedbackModels, feedback)
	}
	if len(reg.Models) > 0 {
		any := firstAlias(sortedAliases(reg.Models))
		if len(reg.Defaults.CodeModels) == 0 {
			reg.Defaults.CodeModels = []string{any}
		}
		if len(reg.Defaults.FeedbackModels) == 0 {
			reg.Defaults.FeedbackModels = []string{any}
		}
	}
	return writeRegistry(filepath.Join(xdgDir, "scienceflow", "models.json"), reg)
}

func (s *ModelStore) activateLocked(alias string) {
	s.active = alias
	s.reg.Defaults.CodeModels = moveToFront(s.reg.Defaults.CodeModels, alias)
	s.reg.Defaults.FeedbackModels = moveToFront(s.reg.Defaults.FeedbackModels, alias)
}

func (s *ModelStore) renameLocked(oldID, newID string) {
	s.reg.Defaults.CodeModels = remapList(s.reg.Defaults.CodeModels, oldID, newID)
	s.reg.Defaults.FeedbackModels = remapList(s.reg.Defaults.FeedbackModels, oldID, newID)
	for sessionID, modelID := range s.sessions {
		if modelID == oldID {
			s.sessions[sessionID] = newID
		}
	}
	for sessionID, modelID := range s.sessionsCoder {
		if modelID == oldID {
			s.sessionsCoder[sessionID] = newID
		}
	}
	for sessionID, modelID := range s.sessionsFeedback {
		if modelID == oldID {
			s.sessionsFeedback[sessionID] = newID
		}
	}
	if s.active == oldID {
		s.active = newID
	}
}

func (s *ModelStore) viewLocked(alias string) ModelView {
	m := s.reg.Models[alias]
	url, key := primaryEndpoint(m)
	return ModelView{
		ID:           alias,
		ModelName:    m.Model,
		APIKey:       key,
		APIKeyMasked: maskSecret(key),
		APIURL:       url,
		Active:       alias == s.active,
	}
}

// primaryEndpoint returns the primary (code[0]) endpoint of a registry model,
// tolerating both the role-map and the plain-list endpoint shapes.
func primaryEndpoint(m registryModel) (string, string) {
	if len(m.Endpoints) == 0 {
		return "", ""
	}
	var roleMap map[string][]registryEndpoint
	if err := json.Unmarshal(m.Endpoints, &roleMap); err == nil && roleMap != nil {
		for _, role := range []string{"code", "feedback"} {
			if list := roleMap[role]; len(list) > 0 {
				return list[0].URL, list[0].Key
			}
		}
		return "", ""
	}
	var plain []registryEndpoint
	if err := json.Unmarshal(m.Endpoints, &plain); err == nil && len(plain) > 0 {
		return plain[0].URL, plain[0].Key
	}
	return "", ""
}

// withPrimaryEndpoint rewrites the primary endpoint of a registry model,
// preserving the rest of the entry (other roles, extra endpoints). An empty
// key keeps the existing one; a missing/unrecognized endpoints field is
// initialized to the role-map shape used by `scienceflow config init`.
func withPrimaryEndpoint(raw json.RawMessage, url, key string) json.RawMessage {
	var roleMap map[string][]registryEndpoint
	if err := json.Unmarshal(raw, &roleMap); err == nil && roleMap != nil {
		list := roleMap["code"]
		if len(list) == 0 {
			roleMap["code"] = []registryEndpoint{{URL: url, Key: key}}
		} else {
			if key == "" {
				key = list[0].Key
			}
			list[0] = registryEndpoint{URL: url, Key: key}
			roleMap["code"] = list
		}
		if b, err := json.Marshal(roleMap); err == nil {
			return b
		}
		return raw
	}
	var plain []registryEndpoint
	if err := json.Unmarshal(raw, &plain); err == nil && plain != nil {
		if len(plain) == 0 {
			plain = []registryEndpoint{{URL: url, Key: key}}
		} else {
			if key == "" {
				key = plain[0].Key
			}
			plain[0] = registryEndpoint{URL: url, Key: key}
		}
		if b, err := json.Marshal(plain); err == nil {
			return b
		}
	}
	endpoint := registryEndpoint{URL: url, Key: key}
	b, err := json.Marshal(map[string][]registryEndpoint{
		"code":     {endpoint},
		"feedback": {endpoint},
	})
	if err != nil {
		return raw
	}
	return b
}

func moveToFront(list []string, alias string) []string {
	out := []string{alias}
	for _, v := range list {
		if v != alias && strings.TrimSpace(v) != "" {
			out = append(out, v)
		}
	}
	return out
}

func removeFromList(list []string, alias string) []string {
	out := make([]string, 0, len(list))
	for _, v := range list {
		if v != alias {
			out = append(out, v)
		}
	}
	return out
}

func remapList(list []string, oldID, newID string) []string {
	out := make([]string, 0, len(list))
	for _, v := range list {
		if v == oldID {
			out = append(out, newID)
		} else {
			out = append(out, v)
		}
	}
	return out
}

func sortedAliases(models map[string]registryModel) []string {
	out := make([]string, 0, len(models))
	for alias := range models {
		out = append(out, alias)
	}
	sort.Strings(out)
	return out
}

func firstAlias(lists ...[]string) string {
	for _, list := range lists {
		for _, v := range list {
			if strings.TrimSpace(v) != "" {
				return v
			}
		}
	}
	return ""
}

func maskSecret(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if len(s) <= 8 {
		return "****"
	}
	return s[:4] + "****" + s[len(s)-4:]
}
