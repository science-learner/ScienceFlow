package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Duration is a time.Duration that unmarshals from either a Go duration
// string ("5s", "250ms") or a bare number (interpreted as seconds).
type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		if s == "" {
			d.Duration = 0
			return nil
		}
		dur, err := time.ParseDuration(s)
		if err != nil {
			return fmt.Errorf("invalid duration %q: %w", s, err)
		}
		d.Duration = dur
		return nil
	}
	var n int64
	if err := json.Unmarshal(b, &n); err != nil {
		return fmt.Errorf("duration must be a string like \"5s\" or a number of seconds")
	}
	d.Duration = time.Duration(n) * time.Second
	return nil
}

type Config struct {
	Server    ServerConfig    `json:"server"`
	Auth      AuthConfig      `json:"auth"`
	Backend   BackendConfig   `json:"backend"`
	Agent     AgentConfig     `json:"agent"`
	Workspace WorkspaceConfig `json:"workspace"`
	Inputs    []InputConfig   `json:"inputs"`
	Registry  RegistryConfig  `json:"registry"`
	Scan      ScanConfig      `json:"scan"`
}

type BackendConfig struct {
	URL string `json:"url"`
}

// AgentConfig controls how the gateway launches ScienceFlow agent tasks per
// user/session. The gateway does NOT modify the agent codebase; it shells out
// to `python -m scienceflow.cli` with a per-session manifest, placing the
// execution workspace under $SCIFLOW_WORKSPACE_ROOT/<user>/<session>.
//
// Module resolution: when RepoRoot is set, the subprocess runs with cwd there
// and the in-repo scienceflow package shadows site-packages (repo mode). When
// RepoRoot is empty, cwd falls back to the task workspace and `python -m
// scienceflow.cli` resolves the pip-installed package from site-packages
// (installed mode).
type AgentConfig struct {
	Enabled          bool     `json:"enabled"`            // master switch for agent invocation
	Python           string   `json:"python"`             // python interpreter (or "uv run python")
	Module           string   `json:"module"`             // CLI module, default "scienceflow.cli"
	Command          string   `json:"command"`            // subcommand: repl | run
	ConfigYAML       string   `json:"config_yaml"`        // -c config path passed to the CLI; empty = installed package's built-in default
	RepoRoot         string   `json:"repo_root"`          // cwd for the subprocess (ScienceFlow repo root); empty = task workspace (installed mode)
	WorkspaceRoot    string   `json:"workspace_root"`     // overrides $SCIFLOW_WORKSPACE_ROOT if non-empty
	InputDataDir     string   `json:"input_data_dir"`     // optional -d shared dataset root
	ExpID            string   `json:"exp_id"`             // optional --exp-id
	Timeout          Duration `json:"timeout"`            // per-invocation wall-clock cap; 0 = unbounded
	HeavyTimeout     Duration `json:"heavy_timeout"`      // wall-clock cap for heavy (run --type lnr) tasks; 0 = use timeout
	MaxConcurrent    int      `json:"max_concurrent"`     // max simultaneous running agent processes (0 = unlimited)
	MaxQueue         int      `json:"max_queue"`          // max tasks waiting in queue when all slots full; 0 = unlimited
	FileSyncInterval Duration `json:"file_sync_interval"` // workspace file-tree change polling interval
	LogDirName       string   `json:"log_dir_name"`       // subdir under workspace for agent logs fed back via SSE
	ModelStorePath   string   `json:"model_store_path"`   // JSON file for runtime model configs
	Timezone         string   `json:"timezone"`           // IANA zone for human-facing timestamps (e.g. stop markers); container TZ is unreliable
}

type AuthConfig struct {
	Enabled     bool     `json:"enabled"`
	File        string   `json:"file"`
	TokenSecret string   `json:"token_secret"`
	TokenTTL    Duration `json:"token_ttl"`
	SessionTTL  Duration `json:"session_ttl"`
}

// WorkspaceConfig limits the file/directory transfer endpoints.
type WorkspaceConfig struct {
	UploadMaxBytes  int64 `json:"upload_max_bytes"`  // multipart file-upload request body cap
	ArchiveMaxBytes int64 `json:"archive_max_bytes"` // zip archive upload body cap
	ExtractMaxFiles int   `json:"extract_max_files"` // max entries extracted from one zip
	ExtractMaxBytes int64 `json:"extract_max_bytes"` // max total bytes extracted from one zip
	ZipMaxEntries   int   `json:"zip_max_entries"`   // max entries packed into one directory download
	ZipMaxBytes     int64 `json:"zip_max_bytes"`     // max total bytes packed into one directory download
}

type ServerConfig struct {
	Host    string `json:"host"`
	Port    int    `json:"port"`
	SSEPath string `json:"sse_path"`
}

type MultilineConfig struct {
	Pattern string   `json:"pattern"`
	Negate  bool     `json:"negate"`
	Match   string   `json:"match"` // after | before
	Timeout Duration `json:"timeout"`
}

type InputConfig struct {
	Name          string          `json:"name"`
	Path          string          `json:"path"`
	Mode          string          `json:"mode"`           // line | raw | prefix (default line)
	PrefixFormat  string          `json:"prefix_format"`  // strftime, required when mode=prefix
	PrefixTimeout Duration        `json:"prefix_timeout"` // prefix 模式的挂起刷新超时
	TailFiles     bool            `json:"tail_files"`
	IgnoreOlder   Duration        `json:"ignore_older"`
	PollInterval  Duration        `json:"poll_interval"`
	Multiline     MultilineConfig `json:"multiline"` // 仅 mode=line 使用
	// StartOffset pins the read position to the file size captured when
	// WatchFile registered a dynamically-tracked file (e.g. a task's RAW.log).
	// Without it the harvester would tail from whatever size the file has when
	// it attaches, silently skipping bytes written in between (task header,
	// early output). HasStartOffset distinguishes "start at 0" from "unset".
	StartOffset    int64 `json:"start_offset,omitempty"`
	HasStartOffset bool  `json:"has_start_offset,omitempty"`
}

type RegistryConfig struct {
	Path          string   `json:"path"`
	FlushInterval Duration `json:"flush_interval"`
}

type ScanConfig struct {
	Frequency Duration `json:"frequency"`
}

// DefaultModelStorePath returns the model registry location the installed
// scienceflow CLI resolves (foundation.config.llm.model_registry.
// default_model_config_path): $XDG_CONFIG_HOME/scienceflow/models.json, or
// ~/.config/scienceflow/models.json. Managing the same file is what makes the
// gateway's /models endpoints effective for agent runs.
func DefaultModelStorePath() string {
	if xdg := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME")); xdg != "" {
		return filepath.Join(xdg, "scienceflow", "models.json")
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.Join(home, ".config", "scienceflow", "models.json")
	}
	return "data/models.json"
}

func Load(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	c.applyDefaults()
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &c, nil
}

func (c *Config) applyDefaults() {
	if c.Server.Host == "" {
		c.Server.Host = "0.0.0.0"
	}
	if c.Server.Port == 0 {
		c.Server.Port = 8080
	}
	if c.Server.SSEPath == "" {
		c.Server.SSEPath = "/events"
	}
	if c.Backend.URL == "" {
		c.Backend.URL = "http://127.0.0.1:8200"
	}
	if c.Agent.Module == "" {
		c.Agent.Module = "scienceflow.cli"
	}
	if c.Agent.Command == "" {
		c.Agent.Command = "repl"
	}
	if c.Agent.Python == "" {
		c.Agent.Python = "python"
	}
	if c.Agent.LogDirName == "" {
		c.Agent.LogDirName = "task_logs"
	}
	if c.Agent.ModelStorePath == "" {
		c.Agent.ModelStorePath = DefaultModelStorePath()
	}
	if c.Agent.MaxConcurrent == 0 {
		c.Agent.MaxConcurrent = 4
	}
	if c.Agent.MaxQueue == 0 {
		c.Agent.MaxQueue = 16
	}
	if c.Agent.FileSyncInterval.Duration == 0 {
		c.Agent.FileSyncInterval.Duration = 3 * time.Second
	}
	if c.Workspace.UploadMaxBytes == 0 {
		c.Workspace.UploadMaxBytes = 100 << 20
	}
	if c.Workspace.ArchiveMaxBytes == 0 {
		c.Workspace.ArchiveMaxBytes = 512 << 20
	}
	if c.Workspace.ExtractMaxFiles == 0 {
		c.Workspace.ExtractMaxFiles = 10000
	}
	if c.Workspace.ExtractMaxBytes == 0 {
		c.Workspace.ExtractMaxBytes = 2 << 30
	}
	if c.Workspace.ZipMaxEntries == 0 {
		c.Workspace.ZipMaxEntries = 20000
	}
	if c.Workspace.ZipMaxBytes == 0 {
		c.Workspace.ZipMaxBytes = 2 << 30
	}
	if c.Auth.File == "" {
		c.Auth.File = "AUTH.yaml"
	}
	if c.Agent.Timezone == "" {
		c.Agent.Timezone = "Asia/Shanghai"
	}
	if c.Auth.TokenTTL.Duration == 0 {
		c.Auth.TokenTTL.Duration = 24 * time.Hour
	}
	if c.Auth.SessionTTL.Duration == 0 {
		c.Auth.SessionTTL.Duration = 1 * time.Hour
	}
	if c.Registry.Path == "" {
		c.Registry.Path = "data/registry.json"
	}
	if c.Registry.FlushInterval.Duration == 0 {
		c.Registry.FlushInterval.Duration = 5 * time.Second
	}
	if c.Scan.Frequency.Duration == 0 {
		c.Scan.Frequency.Duration = 10 * time.Second
	}
	for i := range c.Inputs {
		if c.Inputs[i].PollInterval.Duration == 0 {
			c.Inputs[i].PollInterval.Duration = 250 * time.Millisecond
		}
		if c.Inputs[i].Mode == "" {
			c.Inputs[i].Mode = "line"
		}
		if c.Inputs[i].Mode == "prefix" && c.Inputs[i].PrefixTimeout.Duration == 0 {
			c.Inputs[i].PrefixTimeout.Duration = 50 * time.Millisecond
		}
		if c.Inputs[i].Multiline.Match == "" {
			c.Inputs[i].Multiline.Match = "after"
		}
		if c.Inputs[i].Multiline.Timeout.Duration == 0 {
			c.Inputs[i].Multiline.Timeout.Duration = 5 * time.Second
		}
	}
}

func (c *Config) validate() error {
	if len(c.Inputs) == 0 {
		return fmt.Errorf("at least one input is required")
	}
	for i, in := range c.Inputs {
		if in.Path == "" {
			return fmt.Errorf("inputs[%d].path is required", i)
		}
		switch in.Mode {
		case "", "line", "raw", "prefix":
		default:
			return fmt.Errorf("inputs[%d].mode must be one of \"line\", \"raw\", \"prefix\"", i)
		}
		if in.Mode == "prefix" && in.PrefixFormat == "" {
			return fmt.Errorf("inputs[%d].prefix_format is required when mode=prefix", i)
		}
		switch in.Multiline.Match {
		case "", "after", "before":
		default:
			return fmt.Errorf("inputs[%d].multiline.match must be \"after\" or \"before\"", i)
		}
	}
	if c.Agent.Enabled {
		// repo_root is optional: empty means installed-package mode where the
		// subprocess cwd falls back to the task workspace and the CLI module
		// resolves from site-packages.
		switch c.Agent.Command {
		case "repl", "run":
		default:
			return fmt.Errorf("agent.command must be \"repl\" or \"run\"")
		}
	}
	return nil
}
