package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func newTestStore(t *testing.T, path string) *ModelStore {
	t.Helper()
	s, err := NewModelStore(path)
	if err != nil {
		t.Fatalf("NewModelStore: %v", err)
	}
	return s
}

func mustCreate(t *testing.T, s *ModelStore, name, url, key string) ModelView {
	t.Helper()
	m, err := s.Create(ModelCreateRequest{ModelName: name, APIURL: url, APIKey: key})
	if err != nil {
		t.Fatalf("create %s: %v", name, err)
	}
	return m
}

func TestModelStoreCRUDRegistrySchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scienceflow", "models.json")
	s := newTestStore(t, path)

	m := mustCreate(t, s, "glm-5", "https://api.example/v1", "key-1")
	if m.ID != "glm-5" || !m.Active || m.APIURL != "https://api.example/v1" {
		t.Fatalf("unexpected create view: %+v", m)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("registry file missing: %v", err)
	}
	if fi, _ := os.Stat(path); fi.Mode().Perm() != 0o600 {
		t.Fatalf("registry perms = %v, want 0600", fi.Mode().Perm())
	}
	var reg registryFile
	if err := json.Unmarshal(b, &reg); err != nil {
		t.Fatalf("registry parse: %v", err)
	}
	if reg.Version != 1 || len(reg.Models) != 1 {
		t.Fatalf("unexpected registry: %+v", reg)
	}
	spec := reg.Models["glm-5"]
	if spec.Model != "glm-5" {
		t.Fatalf("unexpected model spec: %+v", spec)
	}
	var ep map[string][]registryEndpoint
	if err := json.Unmarshal(spec.Endpoints, &ep); err != nil {
		t.Fatalf("endpoints shape: %v", err)
	}
	if len(ep["code"]) != 1 || ep["code"][0].URL != "https://api.example/v1" || ep["code"][0].Key != "key-1" {
		t.Fatalf("unexpected code endpoint: %+v", ep)
	}
	if len(reg.Defaults.CodeModels) != 1 || reg.Defaults.CodeModels[0] != "glm-5" ||
		len(reg.Defaults.FeedbackModels) != 1 || reg.Defaults.FeedbackModels[0] != "glm-5" {
		t.Fatalf("unexpected defaults: %+v", reg.Defaults)
	}

	mustCreate(t, s, "glm-air", "https://air/v1", "k2")
	if _, err := s.Activate("glm-air"); err != nil {
		t.Fatalf("activate: %v", err)
	}

	s2 := newTestStore(t, path)
	got := s2.List()
	if len(got) != 2 || got[0].ID != "glm-5" || got[1].ID != "glm-air" {
		t.Fatalf("unexpected list: %+v", got)
	}
	if got[0].Active || !got[1].Active {
		t.Fatalf("active flags wrong: %+v", got)
	}
	if masked := got[1].APIKeyMasked; masked == "k2" || masked == "" {
		t.Fatalf("mask broken: %q", masked)
	}

	u, err := s2.Update("glm-air", ModelUpdateRequest{ModelName: "glm-air", APIURL: "https://air2/v1"})
	if err != nil {
		t.Fatalf("update: %v", err)
	}
	if u.APIURL != "https://air2/v1" {
		t.Fatalf("update url not applied: %+v", u)
	}
	rn, err := s2.Update("glm-air", ModelUpdateRequest{ModelName: "glm-air-2", APIURL: "https://air2/v1"})
	if err != nil || rn.ID != "glm-air-2" {
		t.Fatalf("rename: %v %+v", err, rn)
	}

	s3 := newTestStore(t, path)
	var reg2 registryFile
	b2, _ := os.ReadFile(path)
	_ = json.Unmarshal(b2, &reg2)
	if reg2.Defaults.CodeModels[0] != "glm-air-2" || reg2.Models["glm-air-2"].Model != "glm-air-2" {
		t.Fatalf("rename not remapped in defaults: %+v", reg2.Defaults)
	}
	if err := s3.Delete("glm-air-2"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	lst := newTestStore(t, path).List()
	if len(lst) != 1 || lst[0].ID != "glm-5" || !lst[0].Active {
		t.Fatalf("post-delete list wrong: %+v", lst)
	}
}

func TestDeleteLastModelRemovesFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.json")
	s := newTestStore(t, path)
	mustCreate(t, s, "only", "https://x/v1", "k")
	if err := s.Delete("only"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("registry file should be removed when empty, err=%v", err)
	}
}

func TestSessionStageAliasesAndRegistryMaterialization(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.json")
	s := newTestStore(t, path)
	mustCreate(t, s, "m-a", "https://a/v1", "ka")
	mustCreate(t, s, "m-b", "https://b/v1", "kb")

	if _, _, ok := s.SessionStageAliases("sess-1"); ok {
		t.Fatal("no overrides should resolve to global defaults")
	}

	if err := s.SetSessionStages("sess-1", "m-b", "m-a"); err != nil {
		t.Fatalf("set stages: %v", err)
	}
	code, feedback, ok := s.SessionStageAliases("sess-1")
	if !ok || code != "m-b" || feedback != "m-a" {
		t.Fatalf("stage aliases: %q %q %v", code, feedback, ok)
	}

	xdg := t.TempDir()
	if err := s.WriteSessionRegistry(xdg, code, feedback); err != nil {
		t.Fatalf("write session registry: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(xdg, "scienceflow", "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	var reg registryFile
	if err := json.Unmarshal(b, &reg); err != nil {
		t.Fatalf("session registry parse: %v", err)
	}
	if reg.Defaults.CodeModels[0] != "m-b" || reg.Defaults.FeedbackModels[0] != "m-a" {
		t.Fatalf("session defaults wrong: %+v", reg.Defaults)
	}
	if len(reg.Models) != 2 {
		t.Fatalf("session registry must carry all models: %d", len(reg.Models))
	}

	if err := s.SetSessionModel("sess-2", "m-b"); err != nil {
		t.Fatalf("set session model: %v", err)
	}
	code, feedback, ok = s.SessionStageAliases("sess-2")
	if !ok || code != "m-b" || feedback != "m-b" {
		t.Fatalf("session main model override: %q %q %v", code, feedback, ok)
	}
}
