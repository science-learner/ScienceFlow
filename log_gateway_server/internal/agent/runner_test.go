package agent

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"log_gateway_server/internal/config"
)

func TestBuildArgsHeavyUsesParallelManifest(t *testing.T) {
	r := &Runner{cfg: config.AgentConfig{Module: "scienceflow.cli"}}
	task := &Task{
		Mode:      ModeHeavy,
		Query:     "build a model",
		Workspace: "/ws/admin/s1/run",
		manifest:  "/ws/admin/s1/gateway_manifest_heavy.yaml",
	}

	got := strings.Join(r.buildArgs(task), " ")
	want := "-m scienceflow.cli parallel -m /ws/admin/s1/gateway_manifest_heavy.yaml -j 1"
	if got != want {
		t.Errorf("heavy args = %q, want %q", got, want)
	}
	if strings.Contains(got, "--type") || strings.Contains(got, " -t ") {
		t.Errorf("heavy args must not use the removed --type/-t options: %q", got)
	}
}

func TestBuildArgsLiteUnchanged(t *testing.T) {
	r := &Runner{cfg: config.AgentConfig{Module: "scienceflow.cli", Command: "repl"}}
	task := &Task{
		Mode:      ModeLite,
		Query:     "q",
		Workspace: "/ws/admin/s1/run",
		manifest:  "/ws/admin/s1/gateway_manifest.yaml",
	}
	got := strings.Join(r.buildArgs(task), " ")
	want := "-m scienceflow.cli repl -m /ws/admin/s1/gateway_manifest.yaml --auto-first-user --exit-after-auto --plain"
	if got != want {
		t.Errorf("lite args = %q, want %q", got, want)
	}
}

func TestWriteHeavyManifest(t *testing.T) {
	root := t.TempDir()
	ws := filepath.Join(root, "admin", "s1")
	if err := os.MkdirAll(ws, 0o755); err != nil {
		t.Fatalf("mkdir ws: %v", err)
	}

	r := &Runner{cfg: config.AgentConfig{
		Module:       "scienceflow.cli",
		InputDataDir: "/data/input",
		ConfigYAML:   "scienceflow/config/default.yaml",
	}}
	path, err := r.writeHeavyManifest(ws, "admin", "s1", "long horizon task")
	if err != nil {
		t.Fatalf("writeHeavyManifest: %v", err)
	}
	if filepath.Dir(path) != ws {
		t.Errorf("manifest dir = %s, want %s", filepath.Dir(path), ws)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var doc struct {
		Defaults map[string]any `yaml:"defaults"`
		Tasks    []struct {
			ExpID        string `yaml:"exp_id"`
			RunID        string `yaml:"run_id"`
			Task         string `yaml:"task"`
			InputDataDir string `yaml:"input_data_dir"`
		} `yaml:"tasks"`
	}
	if err := yaml.Unmarshal(b, &doc); err != nil {
		t.Fatalf("parse manifest: %v\n%s", err, b)
	}
	if doc.Defaults["type"] != "lnr" {
		t.Errorf("defaults.type = %v, want lnr", doc.Defaults["type"])
	}
	// workspace_base/run_id/exp_id must resolve back to the session run dir.
	if got, want := doc.Defaults["workspace_base"], filepath.Join(root, "admin"); got != want {
		t.Errorf("defaults.workspace_base = %v, want %s", got, want)
	}
	if doc.Defaults["config"] != "scienceflow/config/default.yaml" {
		t.Errorf("defaults.config = %v", doc.Defaults["config"])
	}
	if len(doc.Tasks) != 1 {
		t.Fatalf("tasks = %d, want 1", len(doc.Tasks))
	}
	tk := doc.Tasks[0]
	if tk.ExpID != "run" || tk.RunID != "s1" || tk.Task != "long horizon task" || tk.InputDataDir != "/data/input" {
		t.Errorf("task = %+v", tk)
	}
}

func TestHeavyLogPath(t *testing.T) {
	heavy := &Task{Mode: ModeHeavy, Workspace: "/ws/admin/s1/run"}
	if got, want := heavy.HeavyLogPath(), "/ws/admin/s1/run/task_logs/parallel_subprocess.log"; got != want {
		t.Errorf("HeavyLogPath = %q, want %q", got, want)
	}
	lite := &Task{Mode: ModeLite, Workspace: "/ws/admin/s1/run"}
	if got := lite.HeavyLogPath(); got != "" {
		t.Errorf("lite HeavyLogPath = %q, want empty", got)
	}
}
