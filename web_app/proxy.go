package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// ProxyRequest is a generic HTTP request forwarded to the ScienceFlow gateway.
type ProxyRequest struct {
	Method  string            `json:"method"`
	Path    string            `json:"path"`
	Body    string            `json:"body,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
}

// TransportMeta mirrors the X-SF-Transport-* headers sent by the backend.
type TransportMeta struct {
	Mode         string `json:"mode,omitempty"`
	Cache        string `json:"cache,omitempty"`
	BuildMs      int64  `json:"build_ms,omitempty"`
	ResponseMs   int64  `json:"response_ms,omitempty"`
	PayloadBytes int64  `json:"payload_bytes,omitempty"`
	ModulesCount int    `json:"modules_count,omitempty"`
}

// ProxyResponse is the generic HTTP response returned to the frontend.
type ProxyResponse struct {
	Status      int            `json:"status"`
	Body        string         `json:"body"`
	ETag        string         `json:"etag,omitempty"`
	NotModified bool           `json:"notModified,omitempty"`
	Transport   *TransportMeta `json:"transport,omitempty"`
}

// UploadedFile is a single file payload (base64-encoded) for dataset upload.
type UploadedFile struct {
	Filename string `json:"filename"`
	Data     string `json:"data"`
}

// UploadResult mirrors the backend's dataset upload response.
type UploadResult struct {
	Path string `json:"path"`
	Name string `json:"name"`
}

// WorkspaceUploadResult mirrors the gateway's file upload response.
type WorkspaceUploadResult struct {
	Root     string                `json:"root"`
	Uploaded []WorkspaceUploadItem `json:"uploaded"`
}

type WorkspaceUploadItem struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// WorkspaceDirUploadResult mirrors the gateway's directory upload response.
type WorkspaceDirUploadResult struct {
	Root            string `json:"root"`
	Extracted       int    `json:"extracted"`
	TotalBytes      int64  `json:"total_bytes"`
	SkippedSymlinks int    `json:"skipped_symlinks"`
}

// FilePayload carries a binary download back to the frontend as base64.
type FilePayload struct {
	Filename    string `json:"filename"`
	ContentType string `json:"contentType"`
	Data        string `json:"data"`
}

// Proxy forwards an arbitrary HTTP request to the log gateway and returns the
// raw response body. It is used for all JSON/text endpoints (the gateway
// reverse-proxies /api/* to the ScienceFlow backend when needed).
func (a *App) Proxy(req ProxyRequest) (ProxyResponse, error) {
	method := strings.ToUpper(strings.TrimSpace(req.Method))
	if method == "" {
		method = http.MethodGet
	}

	path := req.Path
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	var bodyReader io.Reader
	if req.Body != "" {
		bodyReader = bytes.NewReader([]byte(req.Body))
	}

	httpReq, err := http.NewRequest(method, a.gatewayURL+path, bodyReader)
	if err != nil {
		return ProxyResponse{}, err
	}
	for k, v := range req.Headers {
		httpReq.Header.Set(k, v)
	}
	if httpReq.Header.Get("Content-Type") == "" && req.Body != "" {
		httpReq.Header.Set("Content-Type", "application/json")
	}

	start := time.Now()
	resp, err := (&http.Client{}).Do(httpReq)
	if err != nil {
		return ProxyResponse{}, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ProxyResponse{}, err
	}

	out := ProxyResponse{
		Status:      resp.StatusCode,
		Body:        string(raw),
		ETag:        resp.Header.Get("ETag"),
		NotModified: resp.StatusCode == http.StatusNotModified,
	}
	out.Transport = extractTransportMeta(resp.Header, time.Since(start), len(raw))
	return out, nil
}

// DownloadZip proxies the workspace download-zip endpoint and returns the
// resulting archive as base64.
func (a *App) DownloadZip(paths []string, taskRoot string, token string) (FilePayload, error) {
	body, err := json.Marshal(map[string][]string{"paths": paths})
	if err != nil {
		return FilePayload{}, err
	}

	target := a.gatewayURL + "/api/workspace/files/download-zip"
	if taskRoot != "" {
		target += "?task_root=" + url.QueryEscape(taskRoot)
	}

	req, err := http.NewRequest(http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return FilePayload{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return FilePayload{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return FilePayload{}, fmt.Errorf("download failed: %d", resp.StatusCode)
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return FilePayload{}, err
	}
	return FilePayload{
		Filename:    "workspace.zip",
		ContentType: resp.Header.Get("Content-Type"),
		Data:        base64.StdEncoding.EncodeToString(raw),
	}, nil
}

// UploadDataset uploads a single dataset file (multipart field "file").
func (a *App) UploadDataset(file UploadedFile, sessionID string, token string) (UploadResult, error) {
	return a.uploadFiles("file", []UploadedFile{file}, sessionID, token)
}

// UploadDatasetFolder uploads multiple dataset files (multipart field "files").
func (a *App) UploadDatasetFolder(files []UploadedFile, sessionID string, token string) (UploadResult, error) {
	return a.uploadFiles("files", files, sessionID, token)
}

// ── Gateway-native workspace file/directory upload/download ──
// These call the gateway_server's own endpoints (not the backend reverse proxy):
//   POST /api/workspace/file/upload
//   POST /api/workspace/directory/upload
//   GET  /api/workspace/file/download
//   GET  /api/workspace/directory/download
// All use session_id + gateway token authentication.

// WorkspaceDownload downloads a single file or a directory (as zip) from the
// gateway. If isDir is true, uses /api/workspace/directory/download; otherwise
// /api/workspace/file/download. The response is base64-encoded binary.
func (a *App) WorkspaceDownload(token string, sessionID string, path string, isDir bool) (FilePayload, error) {
	var endpoint string
	if isDir {
		endpoint = a.gatewayURL + "/api/workspace/directory/download"
	} else {
		endpoint = a.gatewayURL + "/api/workspace/file/download"
	}
	params := url.Values{}
	params.Set("session_id", sessionID)
	params.Set("path", path)
	endpoint += "?" + params.Encode()

	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return FilePayload{}, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return FilePayload{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return FilePayload{}, fmt.Errorf("download failed: %d %s", resp.StatusCode, string(raw))
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return FilePayload{}, err
	}

	filename := path
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		filename = path[idx+1:]
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	if isDir {
		filename = filename + ".zip"
		contentType = "application/zip"
	}

	return FilePayload{
		Filename:    filename,
		ContentType: contentType,
		Data:        base64.StdEncoding.EncodeToString(raw),
	}, nil
}

// WorkspaceUploadFiles uploads one or more files to a target directory in the
// workspace via multipart form-data. targetDir is a workspace-root-relative
// path (empty = root). Files' Filename may contain sub-path separators.
func (a *App) WorkspaceUploadFiles(token string, sessionID string, targetDir string, files []UploadedFile) (WorkspaceUploadResult, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, f := range files {
		data, err := base64.StdEncoding.DecodeString(f.Data)
		if err != nil {
			return WorkspaceUploadResult{}, err
		}
		part, err := w.CreateFormFile("files", f.Filename)
		if err != nil {
			return WorkspaceUploadResult{}, err
		}
		if _, err := part.Write(data); err != nil {
			return WorkspaceUploadResult{}, err
		}
	}
	if err := w.Close(); err != nil {
		return WorkspaceUploadResult{}, err
	}

	endpoint := a.gatewayURL + "/api/workspace/file/upload"
	params := url.Values{}
	params.Set("session_id", sessionID)
	if targetDir != "" {
		params.Set("path", targetDir)
	}
	endpoint += "?" + params.Encode()

	req, err := http.NewRequest(http.MethodPost, endpoint, &buf)
	if err != nil {
		return WorkspaceUploadResult{}, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return WorkspaceUploadResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return WorkspaceUploadResult{}, fmt.Errorf("upload failed: %d %s", resp.StatusCode, string(raw))
	}

	var result WorkspaceUploadResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return WorkspaceUploadResult{}, err
	}
	return result, nil
}

// WorkspaceUploadZip uploads a zip archive and extracts it to a target directory.
// The zipData is base64-encoded. targetDir is a workspace-root-relative path.
func (a *App) WorkspaceUploadZip(token string, sessionID string, targetDir string, zipData string) (WorkspaceDirUploadResult, error) {
	data, err := base64.StdEncoding.DecodeString(zipData)
	if err != nil {
		return WorkspaceDirUploadResult{}, err
	}

	endpoint := a.gatewayURL + "/api/workspace/directory/upload"
	params := url.Values{}
	params.Set("session_id", sessionID)
	if targetDir != "" {
		params.Set("path", targetDir)
	}
	endpoint += "?" + params.Encode()

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(data))
	if err != nil {
		return WorkspaceDirUploadResult{}, err
	}
	req.Header.Set("Content-Type", "application/zip")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return WorkspaceDirUploadResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return WorkspaceDirUploadResult{}, fmt.Errorf("upload failed: %d %s", resp.StatusCode, string(raw))
	}

	var result WorkspaceDirUploadResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return WorkspaceDirUploadResult{}, err
	}
	return result, nil
}

func (a *App) uploadFiles(field string, files []UploadedFile, sessionID string, token string) (UploadResult, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for _, f := range files {
		data, err := base64.StdEncoding.DecodeString(f.Data)
		if err != nil {
			return UploadResult{}, err
		}
		part, err := w.CreateFormFile(field, f.Filename)
		if err != nil {
			return UploadResult{}, err
		}
		if _, err := part.Write(data); err != nil {
			return UploadResult{}, err
		}
	}
	if err := w.Close(); err != nil {
		return UploadResult{}, err
	}

	endpoint := a.gatewayURL + "/api/datasets/upload"
	if field == "files" {
		endpoint = a.gatewayURL + "/api/datasets/upload-folder"
	}
	if sessionID != "" {
		endpoint += "?session_id=" + url.QueryEscape(sessionID)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, &buf)
	if err != nil {
		return UploadResult{}, err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return UploadResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return UploadResult{}, fmt.Errorf("upload failed: %d %s", resp.StatusCode, string(raw))
	}

	var result UploadResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return UploadResult{}, err
	}
	return result, nil
}

// extractTransportMeta reads the transport diagnostics headers from a response.
func extractTransportMeta(h http.Header, elapsed time.Duration, payloadBytes int) *TransportMeta {
	mode := h.Get("X-SF-Transport-Mode")
	if mode == "" {
		mode = h.Get("X-Transport-Mode")
	}
	cache := h.Get("X-SF-Transport-Cache")
	if cache == "" {
		cache = h.Get("X-Transport-Cache")
	}
	if mode == "" && cache == "" {
		return nil
	}

	toI64 := func(v string) int64 {
		n, _ := strconv.ParseInt(v, 10, 64)
		return n
	}

	return &TransportMeta{
		Mode:         mode,
		Cache:        cache,
		BuildMs:      toI64(firstNonEmpty(h.Get("X-SF-Transport-Build-Ms"), h.Get("X-Transport-Build-Ms"))),
		ResponseMs:   toI64(firstNonEmpty(h.Get("X-SF-Transport-Response-Ms"), h.Get("X-Transport-Response-Ms"))),
		PayloadBytes: toI64(firstNonEmpty(h.Get("X-SF-Transport-Bytes"), h.Get("X-Transport-Bytes"))),
		ModulesCount: int(toI64(firstNonEmpty(h.Get("X-SF-Transport-Modules"), h.Get("X-Transport-Modules")))),
	}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// appConfig is the optional app_config.yaml schema.
type appConfig struct {
	Gateway string `yaml:"gateway"`
}

// loadAppConfig reads app_config.yaml once. The file is looked up next to the
// executable first, then in the working directory. A missing or invalid file
// yields an empty config so callers fall back to their defaults.
func loadAppConfig() appConfig {
	path := locateConfigFile()
	if path == "" {
		return appConfig{}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Printf("app_config: read %s: %v", path, err)
		return appConfig{}
	}
	var cfg appConfig
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		log.Printf("app_config: parse %s: %v", path, err)
		return appConfig{}
	}
	return cfg
}

// resolveGatewayURL resolves the log gateway base URL from app_config.yaml.
func resolveGatewayURL() string {
	return normalizeBaseURL(loadAppConfig().Gateway, defaultGatewayURL)
}

func normalizeBaseURL(raw string, fallback string) string {
	b := strings.TrimRight(strings.TrimSpace(raw), "/")
	if b == "" {
		return fallback
	}
	return b
}

// locateConfigFile finds app_config.yaml next to the executable, then in the
// CWD.
func locateConfigFile() string {
	if exe, err := os.Executable(); err == nil {
		p := filepath.Join(filepath.Dir(exe), "app_config.yaml")
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	if _, err := os.Stat("app_config.yaml"); err == nil {
		return "app_config.yaml"
	}
	return ""
}
