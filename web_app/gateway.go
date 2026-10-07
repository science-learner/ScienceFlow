package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const defaultGatewayURL = "http://127.0.0.1:8080"

// errGatewayUnauthorized is returned when the gateway rejects the token or the
// session (HTTP 401). The frontend uses it to trigger a re-login / session
// rebuild.
var errGatewayUnauthorized = errors.New("用户不存在或密码错误")

// GatewayLoginRequest is the POST /login body.
type GatewayLoginRequest struct {
	UserName string `json:"user_name"`
	Password string `json:"password"`
}

// GatewayLoginResult is the POST /login response.
type GatewayLoginResult struct {
	Token     string   `json:"token"`
	ExpiresAt string   `json:"expires_at"`
	User      string   `json:"user"`
	Sources   []string `json:"sources"`
}

// GatewaySourceInfo is one entry of GET /sources.
type GatewaySourceInfo struct {
	Name  string   `json:"name"`
	Files []string `json:"files"`
}

type GatewayModelInfo struct {
	ID           string `json:"id"`
	ModelName    string `json:"model_name"`
	APIKey       string `json:"api_key"`
	APIKeyMasked string `json:"api_key_masked"`
	APIURL       string `json:"api_url"`
	Active       bool   `json:"is_active"`
	CreatedAt    string `json:"created_at,omitempty"`
	UpdatedAt    string `json:"updated_at,omitempty"`
}

type GatewayModelCreateRequest struct {
	ModelName string `json:"model_name"`
	APIKey    string `json:"api_key"`
	APIURL    string `json:"api_url"`
}

type GatewayModelUpdateRequest struct {
	ModelName string `json:"model_name"`
	APIKey    string `json:"api_key"`
	APIURL    string `json:"api_url"`
}

// GatewayModelsResponse mirrors GET /models.
type GatewayModelsResponse struct {
	Models []GatewayModelInfo `json:"models"`
}

// GatewayHealth reports the result of probing the gateway's GET /healthz.
type GatewayHealth struct {
	Status      string `json:"status"`
	Subscribers int    `json:"subscribers"`
	URL         string `json:"url"`
	LatencyMs   int64  `json:"latency_ms"`
}

type GatewayActivateModelResponse struct {
	ActiveModel GatewayModelInfo `json:"active_model"`
}

// GatewaySession is the sessions* endpoint shape.
type GatewaySession struct {
	SessionID       string               `json:"session_id"`
	Name            string               `json:"name,omitempty"` // 首次提问后的会话名；空表示未命名
	User            string               `json:"user,omitempty"`
	Sources         []string             `json:"sources"`
	LastActive      string               `json:"last_active,omitempty"`
	LastActiveNanos int64                `json:"last_active_nanos,omitempty"`
	Agent           *GatewaySessionAgent `json:"agent,omitempty"`
}

// GatewaySessionAgent mirrors the per-session agent state embedded in the
// unified session view.
type GatewaySessionAgent struct {
	Status string               `json:"status"`
	Task   *GatewayTaskSnapshot `json:"task,omitempty"`
}

// GatewaySessionsResponse mirrors GET /sessions.
type GatewaySessionsResponse struct {
	User     string           `json:"user"`
	Sessions []GatewaySession `json:"sessions"`
}

// GatewaySnapshot mirrors read_cached_content responses (single file or a
// source listing when a source matches multiple files).
type GatewaySnapshot struct {
	Path      string   `json:"path,omitempty"`
	Size      int64    `json:"size,omitempty"`
	Returned  int64    `json:"returned,omitempty"`
	Truncated bool     `json:"truncated,omitempty"`
	Content   string   `json:"content,omitempty"`
	Source    string   `json:"source,omitempty"`
	Files     []string `json:"files,omitempty"`
}

// GatewayLogEvent is the SSE "log" event data forwarded to the frontend.
type GatewayLogEvent struct {
	SessionID string `json:"session_id"`
	Input     string `json:"input"`
	File      string `json:"file"`
	Path      string `json:"path"`
	Message   string `json:"message,omitempty"`
	Data      string `json:"data,omitempty"`
	Encoding  string `json:"encoding,omitempty"`
	Mode      string `json:"mode,omitempty"`
	Timestamp string `json:"timestamp"`
	Offset    int64  `json:"offset"`
}

// GatewayStatusEvent is the gateway SSE connection state forwarded to the
// frontend: connecting / connected / disconnected / unauthorized.
type GatewayStatusEvent struct {
	Status string `json:"status"`
	Error  string `json:"error,omitempty"`
}

// GatewayBackfillEvent is the SSE "backfill" event: a snapshot of one source's
// cached log content, sent on session switch before live tailing resumes.
type GatewayBackfillEvent struct {
	SessionID string `json:"session_id"`
	Source    string `json:"source"`
	Path      string `json:"path"`
	Size      int64  `json:"size"`
	Returned  int64  `json:"returned"`
	Truncated bool   `json:"truncated"`
	Content   string `json:"content"`
}

// GatewayBackfillDoneEvent is the SSE "backfill-done" marker.
type GatewayBackfillDoneEvent struct {
	SessionID string `json:"session_id"`
	Count     int    `json:"count"`
}

// ── Control-plane methods ──

// gatewayHTTPClient bounds all control-plane requests so a hung gateway
// cannot leave frontend promises pending forever (e.g. session list refresh
// or agent stop would spin "Loading..." / "stopping" indefinitely).
var gatewayHTTPClient = &http.Client{Timeout: 15 * time.Second}

func (a *App) gatewayDo(method, path, token string, body any) (int, []byte, error) {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		rdr = bytes.NewReader(b)
	}

	req, err := http.NewRequest(method, a.gatewayURL+path, rdr)
	if err != nil {
		return 0, nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := gatewayHTTPClient.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, err
	}
	return resp.StatusCode, raw, nil
}

func gatewayHTTPError(status int, raw []byte) error {
	if status == http.StatusUnauthorized {
		return errGatewayUnauthorized
	}
	body := strings.TrimSpace(string(raw))
	if body != "" {
		return fmt.Errorf("gateway: %d %s", status, body)
	}
	return fmt.Errorf("gateway: HTTP %d", status)
}

// GatewayLogin authenticates against the log gateway and returns the token and
// the sources visible to the user.
func (a *App) GatewayLogin(req GatewayLoginRequest) (GatewayLoginResult, error) {
	status, raw, err := a.gatewayDo(http.MethodPost, "/login", "", req)
	if err != nil {
		return GatewayLoginResult{}, err
	}
	if status != http.StatusOK {
		return GatewayLoginResult{}, gatewayHTTPError(status, raw)
	}
	var res GatewayLoginResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayLoginResult{}, err
	}
	return res, nil
}

// GatewayTestConnection probes the gateway server itself (GET /healthz, no
// token required) and reports reachability plus round-trip latency.
func (a *App) GatewayTestConnection() (GatewayHealth, error) {
	start := time.Now()
	status, raw, err := a.gatewayDo(http.MethodGet, "/healthz", "", nil)
	if err != nil {
		return GatewayHealth{}, err
	}
	if status != http.StatusOK {
		return GatewayHealth{}, gatewayHTTPError(status, raw)
	}
	var res GatewayHealth
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayHealth{}, err
	}
	if res.Status == "" {
		res.Status = "ok"
	}
	res.URL = a.gatewayURL
	res.LatencyMs = time.Since(start).Milliseconds()
	return res, nil
}

// GatewayListSources returns the sources visible to the token's user.
func (a *App) GatewayListSources(token string) ([]GatewaySourceInfo, error) {
	status, raw, err := a.gatewayDo(http.MethodGet, "/sources", token, nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, gatewayHTTPError(status, raw)
	}
	var res struct {
		Sources []GatewaySourceInfo `json:"sources"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, err
	}
	return res.Sources, nil
}

func (a *App) GatewayListModels(token string) ([]GatewayModelInfo, error) {
	status, raw, err := a.gatewayDo(http.MethodGet, "/models", token, nil)
	if err != nil {
		return nil, err
	}
	if status != http.StatusOK {
		return nil, gatewayHTTPError(status, raw)
	}
	var res GatewayModelsResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, err
	}
	return res.Models, nil
}

func (a *App) GatewayCreateModel(token string, req GatewayModelCreateRequest) (GatewayModelInfo, error) {
	status, raw, err := a.gatewayDo(http.MethodPost, "/models", token, req)
	if err != nil {
		return GatewayModelInfo{}, err
	}
	if status != http.StatusCreated {
		return GatewayModelInfo{}, gatewayHTTPError(status, raw)
	}
	var res GatewayModelInfo
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayModelInfo{}, err
	}
	return res, nil
}

func (a *App) GatewayUpdateModel(token string, modelID string, req GatewayModelUpdateRequest) (GatewayModelInfo, error) {
	status, raw, err := a.gatewayDo(http.MethodPut, "/models/"+url.PathEscape(modelID), token, req)
	if err != nil {
		return GatewayModelInfo{}, err
	}
	if status != http.StatusOK {
		return GatewayModelInfo{}, gatewayHTTPError(status, raw)
	}
	var res GatewayModelInfo
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayModelInfo{}, err
	}
	return res, nil
}

func (a *App) GatewayDeleteModel(token string, modelID string) error {
	status, raw, err := a.gatewayDo(http.MethodDelete, "/models/"+url.PathEscape(modelID), token, nil)
	if err != nil {
		return err
	}
	if status != http.StatusNoContent {
		return gatewayHTTPError(status, raw)
	}
	return nil
}

func (a *App) GatewayActivateModel(token string, modelID string, sessionID string) (GatewayModelInfo, error) {
	path := "/models/" + url.PathEscape(modelID) + "/activate"
	if sessionID != "" {
		path += "?session_id=" + url.QueryEscape(sessionID)
	}
	status, raw, err := a.gatewayDo(http.MethodPost, path, token, nil)
	if err != nil {
		return GatewayModelInfo{}, err
	}
	if status != http.StatusOK {
		return GatewayModelInfo{}, gatewayHTTPError(status, raw)
	}
	var res GatewayActivateModelResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayModelInfo{}, err
	}
	return res.ActiveModel, nil
}

type GatewayModelStages struct {
	CoderModelID    string `json:"coder_model_id"`
	FeedbackModelID string `json:"feedback_model_id"`
}

func (a *App) GatewayGetModelStages(token string, sessionID string) (GatewayModelStages, error) {
	path := "/models/stages?session_id=" + url.QueryEscape(sessionID)
	status, raw, err := a.gatewayDo(http.MethodGet, path, token, nil)
	if err != nil {
		return GatewayModelStages{}, err
	}
	if status != http.StatusOK {
		return GatewayModelStages{}, gatewayHTTPError(status, raw)
	}
	var res GatewayModelStages
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayModelStages{}, err
	}
	return res, nil
}

func (a *App) GatewaySetModelStages(token string, sessionID string, coderModelID string, feedbackModelID string) (GatewayModelStages, error) {
	body := map[string]string{
		"session_id":        sessionID,
		"coder_model_id":    coderModelID,
		"feedback_model_id": feedbackModelID,
	}
	status, raw, err := a.gatewayDo(http.MethodPut, "/models/stages", token, body)
	if err != nil {
		return GatewayModelStages{}, err
	}
	if status != http.StatusOK {
		return GatewayModelStages{}, gatewayHTTPError(status, raw)
	}
	var res GatewayModelStages
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayModelStages{}, err
	}
	return res, nil
}

// GatewayCreateSession creates a subscription session for the token's user.
func (a *App) GatewayCreateSession(token string, sources []string) (GatewaySession, error) {
	status, raw, err := a.gatewayDo(http.MethodPost, "/sessions", token, map[string][]string{"sources": sources})
	if err != nil {
		return GatewaySession{}, err
	}
	if status != http.StatusOK {
		return GatewaySession{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySession
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySession{}, err
	}
	return res, nil
}

// GatewayGetSession queries a session.
func (a *App) GatewayGetSession(token string, sessionID string) (GatewaySession, error) {
	status, raw, err := a.gatewayDo(http.MethodGet, "/sessions/"+url.PathEscape(sessionID), token, nil)
	if err != nil {
		return GatewaySession{}, err
	}
	if status != http.StatusOK {
		return GatewaySession{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySession
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySession{}, err
	}
	return res, nil
}

// GatewayUpdateSessionSources replaces a session's subscriptions (full replace).
func (a *App) GatewayUpdateSessionSources(token string, sessionID string, sources []string) (GatewaySession, error) {
	status, raw, err := a.gatewayDo(http.MethodPut, "/sessions/"+url.PathEscape(sessionID)+"/sources", token, map[string][]string{"sources": sources})
	if err != nil {
		return GatewaySession{}, err
	}
	if status != http.StatusOK {
		return GatewaySession{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySession
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySession{}, err
	}
	return res, nil
}

// GatewayDeleteSession removes a session.
func (a *App) GatewayDeleteSession(token string, sessionID string) error {
	status, raw, err := a.gatewayDo(http.MethodDelete, "/sessions/"+url.PathEscape(sessionID), token, nil)
	if err != nil {
		return err
	}
	if status == http.StatusNoContent {
		return nil
	}
	return gatewayHTTPError(status, raw)
}

// GatewayListSessions lists all live sessions owned by the token's user.
func (a *App) GatewayListSessions(token string) (GatewaySessionsResponse, error) {
	status, raw, err := a.gatewayDo(http.MethodGet, "/sessions", token, nil)
	if err != nil {
		return GatewaySessionsResponse{}, err
	}
	if status != http.StatusOK {
		return GatewaySessionsResponse{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySessionsResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySessionsResponse{}, err
	}
	return res, nil
}

// GatewayActivateSession activates (switches to) an existing session, refreshing
// its TTL and returning the unified session view.
func (a *App) GatewayActivateSession(token string, sessionID string) (GatewaySession, error) {
	status, raw, err := a.gatewayDo(http.MethodPost, "/sessions/"+url.PathEscape(sessionID)+"/activate", token, nil)
	if err != nil {
		return GatewaySession{}, err
	}
	if status != http.StatusOK {
		return GatewaySession{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySession
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySession{}, err
	}
	return res, nil
}

// GatewayReadSnapshot fetches cached file content by path or by source name.
// Only one of path/source should be set.
func (a *App) GatewayReadSnapshot(token string, path string, source string) (GatewaySnapshot, error) {
	q := ""
	switch {
	case source != "":
		q = "?source=" + url.QueryEscape(source)
	case path != "":
		q = "?path=" + url.QueryEscape(path)
	default:
		return GatewaySnapshot{}, errors.New("path or source is required")
	}
	status, raw, err := a.gatewayDo(http.MethodGet, "/read_cached_content"+q, token, nil)
	if err != nil {
		return GatewaySnapshot{}, err
	}
	if status != http.StatusOK {
		return GatewaySnapshot{}, gatewayHTTPError(status, raw)
	}
	var res GatewaySnapshot
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewaySnapshot{}, err
	}
	return res, nil
}

// ── Agent invoke / status / stop ──

// GatewayInvokeRequest is the POST /sessions/{id}/invoke body.
type GatewayInvokeRequest struct {
	Query string `json:"query"`
	Mode  string `json:"mode,omitempty"` // "lite" | "heavy"
}

// GatewayTaskSnapshot mirrors the task object returned by the agent endpoints.
type GatewayTaskSnapshot struct {
	ID         string `json:"id"`
	User       string `json:"user,omitempty"`
	Session    string `json:"session,omitempty"`
	Status     string `json:"status"`
	Workspace  string `json:"workspace,omitempty"`
	LogDir     string `json:"log_dir,omitempty"`
	RawLogPath string `json:"raw_log_path,omitempty"`
	StartedAt  string `json:"started_at,omitempty"`
	EndedAt    string `json:"ended_at,omitempty"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	Error      string `json:"error,omitempty"`
}

// GatewayQueueStats mirrors the queue_stats field.
type GatewayQueueStats struct {
	Queued  int `json:"queued"`
	Running int `json:"running"`
}

// GatewayAgentStatus mirrors GET /sessions/{id}/agent.
type GatewayAgentStatus struct {
	Task       *GatewayTaskSnapshot `json:"task,omitempty"`
	Output     string               `json:"output,omitempty"`
	Running    bool                 `json:"running,omitempty"`
	QueueStats *GatewayQueueStats   `json:"queue_stats,omitempty"`
	Status     string               `json:"status,omitempty"` // "idle" when no task
}

// GatewayInvokeAgent triggers a ScienceFlow agent task on the session.
func (a *App) GatewayInvokeAgent(token string, sessionID string, req GatewayInvokeRequest) (GatewayTaskSnapshot, error) {
	body := req
	if req.Mode != "" {
		// Embed mode into query so the gateway can pass it to the agent.
		body.Query = fmt.Sprintf("[mode=%s] %s", req.Mode, req.Query)
	}
	status, raw, err := a.gatewayDo(http.MethodPost, "/sessions/"+url.PathEscape(sessionID)+"/invoke", token, body)
	if err != nil {
		return GatewayTaskSnapshot{}, err
	}
	if status == http.StatusConflict {
		return GatewayTaskSnapshot{}, errors.New("an agent task is already running for this session")
	}
	if status == http.StatusServiceUnavailable {
		return GatewayTaskSnapshot{}, errors.New("agent task queue is full; try again later")
	}
	if status != http.StatusAccepted {
		return GatewayTaskSnapshot{}, gatewayHTTPError(status, raw)
	}
	var res GatewayTaskSnapshot
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayTaskSnapshot{}, err
	}
	return res, nil
}

// GatewayAgentStatus queries the current agent task status and output.
func (a *App) GatewayAgentStatus(token string, sessionID string) (GatewayAgentStatus, error) {
	status, raw, err := a.gatewayDo(http.MethodGet, "/sessions/"+url.PathEscape(sessionID)+"/agent", token, nil)
	if err != nil {
		return GatewayAgentStatus{}, err
	}
	if status != http.StatusOK {
		return GatewayAgentStatus{}, gatewayHTTPError(status, raw)
	}
	var res GatewayAgentStatus
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayAgentStatus{}, err
	}
	return res, nil
}

// GatewayStopAgent kills the current agent task for the session.
func (a *App) GatewayStopAgent(token string, sessionID string) (GatewayTaskSnapshot, error) {
	status, raw, err := a.gatewayDo(http.MethodDelete, "/sessions/"+url.PathEscape(sessionID)+"/agent", token, nil)
	if err != nil {
		return GatewayTaskSnapshot{}, err
	}
	if status != http.StatusOK {
		return GatewayTaskSnapshot{}, gatewayHTTPError(status, raw)
	}
	var res GatewayTaskSnapshot
	if err := json.Unmarshal(raw, &res); err != nil {
		return GatewayTaskSnapshot{}, err
	}
	return res, nil
}

// ── SSE bridge ──

// gatewayStreamManager owns the single gateway SSE connection and its reconnect
// loop. It mirrors chatStreamManager but subscribes with ?session=<id>.
type gatewayStreamManager struct {
	ctx    context.Context
	mu     sync.Mutex
	cancel context.CancelFunc
	done   chan struct{}
}

func newGatewayStreamManager(ctx context.Context) *gatewayStreamManager {
	return &gatewayStreamManager{ctx: ctx}
}

func (m *gatewayStreamManager) Start(baseURL, sessionID string) {
	m.Stop()
	ctx, cancel := context.WithCancel(m.ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.done = make(chan struct{})
	m.mu.Unlock()
	go m.run(ctx, baseURL, sessionID)
}

func (m *gatewayStreamManager) Stop() {
	m.mu.Lock()
	cancel := m.cancel
	done := m.done
	m.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}

func (m *gatewayStreamManager) run(ctx context.Context, baseURL, sessionID string) {
	defer close(m.done)
	backoff := time.Second
	for {
		if ctx.Err() != nil {
			return
		}
		emitEvent("gateway-status", GatewayStatusEvent{Status: "connecting"})
		err := m.stream(ctx, baseURL, sessionID)
		if ctx.Err() != nil {
			return
		}
		if errors.Is(err, errGatewayUnauthorized) {
			emitEvent("gateway-status", GatewayStatusEvent{Status: "unauthorized"})
			return
		}
		if err != nil {
			emitEvent("gateway-status", GatewayStatusEvent{Status: "disconnected", Error: err.Error()})
		} else {
			backoff = time.Second
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		if backoff < 30*time.Second {
			backoff *= 2
		}
	}
}

func (m *gatewayStreamManager) stream(ctx context.Context, baseURL, sessionID string) error {
	endpoint := baseURL + "/events?session=" + url.QueryEscape(sessionID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		return errGatewayUnauthorized
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gateway stream status %d", resp.StatusCode)
	}

	emitEvent("gateway-status", GatewayStatusEvent{Status: "connected"})
	reader := bufio.NewReader(resp.Body)

	var (
		eventName string
		dataLines []string
	)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return err
		}
		line = strings.TrimRight(line, "\r\n")
		switch {
		case line == "":
			if eventName != "" && len(dataLines) > 0 {
				payload := strings.Join(dataLines, "\n")
				switch eventName {
				case "log":
					m.handleLog(ctx, payload, sessionID)
				case "file":
					m.handleFile(ctx, payload, sessionID)
				case "backfill":
					m.handleBackfill(ctx, payload, sessionID)
				case "backfill-done":
					m.handleBackfillDone(ctx, payload, sessionID)
				}
			}
			eventName, dataLines = "", nil
		case strings.HasPrefix(line, ":"):
			// comment / heartbeat, ignore
		case strings.HasPrefix(line, "event:"):
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

func (m *gatewayStreamManager) handleLog(ctx context.Context, payload string, sessionID string) {
	var evt GatewayLogEvent
	if err := json.Unmarshal([]byte(payload), &evt); err != nil {
		return
	}
	evt.SessionID = sessionID
	emitEvent("gateway-log", evt)
}

func (m *gatewayStreamManager) handleFile(ctx context.Context, payload string, sessionID string) {
	var evt FileTreeEvent
	if err := json.Unmarshal([]byte(payload), &evt); err != nil {
		return
	}
	evt.SessionID = sessionID
	emitEvent("gateway-file", evt)
}

func (m *gatewayStreamManager) handleBackfill(ctx context.Context, payload string, sessionID string) {
	var evt GatewayBackfillEvent
	if err := json.Unmarshal([]byte(payload), &evt); err != nil {
		return
	}
	evt.SessionID = sessionID
	emitEvent("gateway-backfill", evt)
}

func (m *gatewayStreamManager) handleBackfillDone(ctx context.Context, payload string, sessionID string) {
	var evt GatewayBackfillDoneEvent
	if err := json.Unmarshal([]byte(payload), &evt); err != nil {
		return
	}
	evt.SessionID = sessionID
	emitEvent("gateway-backfill-done", evt)
}
