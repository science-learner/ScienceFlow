package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// FileTreeNode mirrors the gateway's file tree node (flat list, relative paths).
type FileTreeNode struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	Type  string `json:"type"`
	Size  int64  `json:"size"`
	Mtime int64  `json:"mtime"`
}

// FileTreeResponse mirrors GET /sessions/{id}/files.
type FileTreeResponse struct {
	Root string         `json:"root"`
	Tree []FileTreeNode `json:"tree"`
}

// FileTreeEvent is the SSE "file" event payload emitted by the gateway.
// kind="tree" carries the full tree; kind="changes" carries deltas.
type FileTreeEvent struct {
	SessionID string         `json:"session_id"`
	Kind      string         `json:"kind"`
	Root      string         `json:"root,omitempty"`
	Tree      []FileTreeNode `json:"tree,omitempty"`
	Added     []FileTreeNode `json:"added,omitempty"`
	Removed   []FileTreeNode `json:"removed,omitempty"`
	Modified  []FileTreeNode `json:"modified,omitempty"`
	Overflow  bool           `json:"overflow,omitempty"`
	Timestamp string         `json:"timestamp,omitempty"`
}

type GatewayMonitorMetrics struct {
	SessionID string                 `json:"session_id"`
	Workspace string                 `json:"workspace"`
	UpdatedAt string                 `json:"updated_at"`
	Sources   []string               `json:"sources"`
	Resources GatewayMonitorResource `json:"resources"`
	Tokens    GatewayMonitorTokens   `json:"tokens"`
	Runtime   GatewayMonitorRuntime  `json:"runtime"`
}

type GatewayMonitorResource struct {
	CPUPercent     float64 `json:"cpu_percent"`
	MemoryPercent  float64 `json:"memory_percent"`
	MemoryUsedGB   float64 `json:"memory_used_gb"`
	StorageBytes   int64   `json:"storage_bytes"`
	StorageFiles   int     `json:"storage_files"`
	StorageDisplay string  `json:"storage_display"`
}

type GatewayMonitorTokens struct {
	Reserved bool  `json:"reserved"`
	Input    int64 `json:"input"`
	Output   int64 `json:"output"`
	Cached   int64 `json:"cached"`
	Total    int64 `json:"total"`
}

type GatewayMonitorRuntime struct {
	Status    string  `json:"status"`
	Elapsed   string  `json:"elapsed"`
	ElapsedMs float64 `json:"elapsed_ms"`
}

func (a *App) GatewayFetchMonitor(token string, sessionID string) (GatewayMonitorMetrics, error) {
	endpoint := a.gatewayURL + "/sessions/" + url.PathEscape(sessionID) + "/monitor"
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return GatewayMonitorMetrics{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return GatewayMonitorMetrics{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		return GatewayMonitorMetrics{}, errGatewayUnauthorized
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return GatewayMonitorMetrics{}, fmt.Errorf("fetch monitor status %d: %s", resp.StatusCode, string(body))
	}
	var result GatewayMonitorMetrics
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return GatewayMonitorMetrics{}, err
	}
	return result, nil
}

// GatewayFetchFiles fetches the current workspace file tree for a session via
// GET /sessions/{id}/files. Used for on-demand full-tree calibration (e.g.
// after an overflow event).
func (a *App) GatewayFetchFiles(token string, sessionID string) (FileTreeResponse, error) {
	endpoint := a.gatewayURL + "/sessions/" + url.PathEscape(sessionID) + "/files"

	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return FileTreeResponse{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return FileTreeResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		return FileTreeResponse{}, errGatewayUnauthorized
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return FileTreeResponse{}, fmt.Errorf("fetch files status %d: %s", resp.StatusCode, string(body))
	}

	var result FileTreeResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return FileTreeResponse{}, err
	}
	return result, nil
}
