package server

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

type sessionMonitorResponse struct {
	SessionID string                 `json:"session_id"`
	Workspace string                 `json:"workspace"`
	UpdatedAt string                 `json:"updated_at"`
	Sources   []string               `json:"sources"`
	Resources sessionMonitorResource `json:"resources"`
	Tokens    sessionMonitorTokens   `json:"tokens"`
	Runtime   sessionMonitorRuntime  `json:"runtime"`
}

type sessionMonitorResource struct {
	CPUPercent     float64 `json:"cpu_percent"`
	MemoryPercent  float64 `json:"memory_percent"`
	MemoryUsedGB   float64 `json:"memory_used_gb"`
	StorageBytes   int64   `json:"storage_bytes"`
	StorageFiles   int     `json:"storage_files"`
	StorageDisplay string  `json:"storage_display"`
}

type sessionMonitorTokens struct {
	Reserved bool  `json:"reserved"`
	Input    int64 `json:"input"`
	Output   int64 `json:"output"`
	Cached   int64 `json:"cached"`
	Total    int64 `json:"total"`
}

type sessionMonitorRuntime struct {
	Status    string  `json:"status"`
	Elapsed   string  `json:"elapsed"`
	ElapsedMs float64 `json:"elapsed_ms"`
}

func (s *Server) handleSessionMonitor(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	id := r.PathValue("id")
	owner := s.sessions.User(id)
	if owner == "" {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if s.auth != nil && owner != user {
		s.writeUnauthorized(w)
		return
	}
	dir, ok := s.sessionWorkspaceDir(id)
	if !ok {
		http.Error(w, "workspace root not configured", http.StatusServiceUnavailable)
		return
	}

	metrics := collectSessionMonitorMetrics(id, dir)
	if s.agent != nil {
		if task := s.agent.Current(id); task != nil {
			snapshot := task.Snapshot()
			if snapshot.Status == "running" && snapshot.PID > 0 {
				cpu, memoryGB, ok := processResources(snapshot.PID)
				if ok {
					metrics.Resources.CPUPercent = cpu
					metrics.Resources.MemoryUsedGB = memoryGB
					metrics.Runtime.Status = string(snapshot.Status)
					metrics.Sources = append(metrics.Sources, "agent process")
					sort.Strings(metrics.Sources)
				}
			}
		}
	}
	writeJSON(w, http.StatusOK, metrics)
}

func collectSessionMonitorMetrics(sessionID, workspace string) sessionMonitorResponse {
	resources := sessionMonitorResource{}
	tokens := sessionMonitorTokens{Reserved: true}
	runtime := sessionMonitorRuntime{Status: "idle"}
	sourceSet := map[string]bool{}
	var newest time.Time

	storageBytes, storageFiles, storageNewest := workspaceStorage(workspace)
	resources.StorageBytes = storageBytes
	resources.StorageFiles = storageFiles
	resources.StorageDisplay = formatBytes(storageBytes)
	newest = maxTime(newest, storageNewest)

	state := loadMonitorState(filepath.Join(workspace, "logs", "monitor_state.json"))
	if len(state) > 0 {
		sourceSet["logs/monitor_state.json"] = true
		resources.CPUPercent = firstFloat(state, "cpu_percent", "cpu", "cpu_usage", "cpu_usage_percent")
		resources.MemoryPercent = firstFloat(state, "memory_percent", "memory", "memory_usage_percent")
		resources.MemoryUsedGB = firstFloat(state, "memory_used_gb", "memory_gb")
		tokens.Input = firstInt(state, "total_tokens_in", "tokens_in", "input_tokens")
		tokens.Output = firstInt(state, "total_tokens_out", "tokens_out", "output_tokens")
		tokens.Cached = firstInt(state, "total_tokens_cached", "tokens_cached", "cached_tokens")
		runtime.Status = firstString(state, "status", "run_status")
		runtime.ElapsedMs = firstFloat(state, "elapsed_ms")
		if runtime.ElapsedMs == 0 {
			runtime.ElapsedMs = firstFloat(state, "elapsed_sec") * 1000
		}
		if runtime.ElapsedMs > 0 {
			runtime.Elapsed = formatDuration(runtime.ElapsedMs)
		}
	}

	resourcePath := filepath.Join(workspace, "task_logs", "resource", "resource_events.jsonl")
	if cpu, memPct, memGB, ok, modTime := latestResourceEvent(resourcePath); ok {
		sourceSet["task_logs/resource/resource_events.jsonl"] = true
		newest = maxTime(newest, modTime)
		if cpu > 0 {
			resources.CPUPercent = cpu
		}
		if memPct > 0 {
			resources.MemoryPercent = memPct
		}
		if memGB > 0 {
			resources.MemoryUsedGB = memGB
		}
	}

	traceTokens, traceNewest, traceFound := summarizeTimeTraces(workspace)
	if traceFound {
		sourceSet["task_logs/scienceflow_time_trace.csv"] = true
		newest = maxTime(newest, traceNewest)
		if traceTokens.Input > tokens.Input {
			tokens.Input = traceTokens.Input
		}
		if traceTokens.Output > tokens.Output {
			tokens.Output = traceTokens.Output
		}
		if traceTokens.Cached > tokens.Cached {
			tokens.Cached = traceTokens.Cached
		}
	}
	tokens.Total = tokens.Input + tokens.Output
	if runtime.Elapsed == "" {
		runtime.Elapsed = "--"
	}
	if runtime.Status == "" {
		runtime.Status = "idle"
	}

	sources := make([]string, 0, len(sourceSet))
	for source := range sourceSet {
		sources = append(sources, source)
	}
	sort.Strings(sources)
	updatedAt := ""
	if !newest.IsZero() {
		updatedAt = newest.UTC().Format(time.RFC3339)
	}
	return sessionMonitorResponse{
		SessionID: sessionID,
		Workspace: workspace,
		UpdatedAt: updatedAt,
		Sources:   sources,
		Resources: resources,
		Tokens:    tokens,
		Runtime:   runtime,
	}
}

func processResources(pid int) (float64, float64, bool) {
	if pid <= 0 {
		return 0, 0, false
	}
	if runtime.GOOS == "windows" {
		return processResourcesWindows(pid)
	}
	return processResourcesUnix(pid)
}

func processResourcesWindows(pid int) (float64, float64, bool) {
	command := exec.Command("powershell", "-NoProfile", "-Command", fmt.Sprintf("$p=Get-Process -Id %d -ErrorAction SilentlyContinue; if ($p) { [Console]::WriteLine(\"{0}|{1}\" -f $p.CPU, $p.WorkingSet64) }", pid))
	output, err := command.Output()
	if err != nil {
		return 0, 0, false
	}
	parts := strings.Split(strings.TrimSpace(string(output)), "|")
	if len(parts) != 2 {
		return 0, 0, false
	}
	cpuSeconds, errCPU := strconv.ParseFloat(strings.TrimSpace(parts[0]), 64)
	memoryBytes, errMemory := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
	if errCPU != nil || errMemory != nil {
		return 0, 0, false
	}
	return cpuSeconds, memoryBytes / (1024 * 1024 * 1024), true
}

func processResourcesUnix(pid int) (float64, float64, bool) {
	command := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "%cpu=,%rss=")
	output, err := command.Output()
	if err != nil {
		return 0, 0, false
	}
	fields := strings.Fields(string(output))
	if len(fields) < 2 {
		return 0, 0, false
	}
	cpu, errCPU := strconv.ParseFloat(fields[0], 64)
	rssKB, errMemory := strconv.ParseFloat(fields[1], 64)
	if errCPU != nil || errMemory != nil {
		return 0, 0, false
	}
	return cpu, rssKB / (1024 * 1024), true
}
func workspaceStorage(root string) (int64, int, time.Time) {
	var total int64
	files := 0
	var newest time.Time
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		total += info.Size()
		files++
		newest = maxTime(newest, info.ModTime())
		return nil
	})
	return total, files, newest
}

func loadMonitorState(path string) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var state map[string]any
	if err := json.Unmarshal(b, &state); err != nil {
		return nil
	}
	return state
}

func latestResourceEvent(path string) (float64, float64, float64, bool, time.Time) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, 0, false, time.Time{}
	}
	defer f.Close()

	var cpu, memPct, memGB float64
	found := false
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event map[string]any
		if json.Unmarshal([]byte(line), &event) != nil {
			continue
		}
		flat := flattenMap(event)
		if v := firstFloat(flat, "cpu_percent", "cpu", "cpu_usage", "cpu_usage_percent"); v > 0 {
			cpu = v
		}
		if v := firstFloat(flat, "memory_percent", "memory", "memory_usage_percent"); v > 0 {
			memPct = v
		}
		if v := firstFloat(flat, "memory_used_gb", "memory_gb"); v > 0 {
			memGB = v
		}
		found = true
	}
	info, _ := os.Stat(path)
	modTime := time.Time{}
	if info != nil {
		modTime = info.ModTime()
	}
	return cpu, memPct, memGB, found, modTime
}

func summarizeTimeTraces(root string) (sessionMonitorTokens, time.Time, bool) {
	var tokens sessionMonitorTokens
	var newest time.Time
	found := false
	paths := []string{filepath.Join(root, "task_logs", "scienceflow_time_trace.csv")}
	matches, _ := filepath.Glob(filepath.Join(root, "task_logs", "workers", "*", "scienceflow_time_trace.csv"))
	paths = append(paths, matches...)
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		reader := csv.NewReader(f)
		reader.FieldsPerRecord = -1
		rows, err := reader.ReadAll()
		_ = f.Close()
		if err != nil || len(rows) < 2 {
			continue
		}
		found = true
		idx := csvIndex(rows[0])
		for _, row := range rows[1:] {
			tokens.Input += csvInt(row, idx, "tokens_input", "input_tokens")
			tokens.Output += csvInt(row, idx, "tokens_output", "output_tokens")
			tokens.Cached += csvInt(row, idx, "tokens_cached", "cached_tokens")
		}
		if info, err := os.Stat(path); err == nil {
			newest = maxTime(newest, info.ModTime())
		}
	}
	tokens.Reserved = true
	tokens.Total = tokens.Input + tokens.Output
	return tokens, newest, found
}

func csvIndex(header []string) map[string]int {
	idx := make(map[string]int, len(header))
	for i, name := range header {
		idx[strings.TrimSpace(name)] = i
	}
	return idx
}

func csvInt(row []string, idx map[string]int, names ...string) int64 {
	for _, name := range names {
		i, ok := idx[name]
		if !ok || i >= len(row) {
			continue
		}
		if v, err := strconv.ParseFloat(strings.TrimSpace(row[i]), 64); err == nil {
			return int64(v)
		}
	}
	return 0
}

func flattenMap(value map[string]any) map[string]any {
	out := map[string]any{}
	var walk func(map[string]any)
	walk = func(m map[string]any) {
		for k, v := range m {
			out[k] = v
			if child, ok := v.(map[string]any); ok {
				walk(child)
			}
		}
	}
	walk(value)
	return out
}

func firstFloat(values map[string]any, keys ...string) float64 {
	for _, key := range keys {
		switch v := values[key].(type) {
		case float64:
			return v
		case int:
			return float64(v)
		case int64:
			return float64(v)
		case string:
			if parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
				return parsed
			}
		}
	}
	return 0
}

func firstInt(values map[string]any, keys ...string) int64 {
	for _, key := range keys {
		switch v := values[key].(type) {
		case float64:
			return int64(v)
		case int:
			return int64(v)
		case int64:
			return v
		case string:
			if parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64); err == nil {
				return int64(parsed)
			}
		}
	}
	return 0
}

func firstString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if text := strings.TrimSpace(toString(values[key])); text != "" {
			return text
		}
	}
	return ""
}

func toString(value any) string {
	if value == nil {
		return ""
	}
	return fmt.Sprint(value)
}

func maxTime(a, b time.Time) time.Time {
	if b.After(a) {
		return b
	}
	return a
}

func formatBytes(bytes int64) string {
	units := []string{"B", "KB", "MB", "GB", "TB"}
	value := float64(bytes)
	unit := 0
	for value >= 1024 && unit < len(units)-1 {
		value /= 1024
		unit++
	}
	if unit == 0 {
		return strconv.FormatInt(bytes, 10) + " B"
	}
	return strconv.FormatFloat(value, 'f', 1, 64) + " " + units[unit]
}

func formatDuration(ms float64) string {
	d := time.Duration(ms) * time.Millisecond
	if d < time.Minute {
		return strconv.Itoa(int(d.Seconds())) + "s"
	}
	if d < time.Hour {
		return strconv.Itoa(int(d.Minutes())) + "m " + strconv.Itoa(int(d.Seconds())%60) + "s"
	}
	return strconv.Itoa(int(d.Hours())) + "h " + strconv.Itoa(int(d.Minutes())%60) + "m"
}
