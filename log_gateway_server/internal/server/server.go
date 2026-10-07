package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"log_gateway_server/internal/agent"
	"log_gateway_server/internal/auth"
	"log_gateway_server/internal/config"
	"log_gateway_server/internal/filewatch"
	"log_gateway_server/internal/hub"
	"log_gateway_server/internal/session"
)

// maxSnapshotBytes caps how much of a file read_cached_content returns.
const maxSnapshotBytes = 16 * 1024 * 1024

// errUnauthorized is the fixed body returned on any auth failure.
const errUnauthorized = "用户不存在或密码错误"

type Server struct {
	hub          *hub.Hub
	host         string
	port         int
	path         string
	logger       *log.Logger
	seq          atomic.Int64
	auth         *auth.Store
	sessions     *session.Manager
	tokenTTL     time.Duration
	backendURL   string
	sourceNames  func() []string
	sourceOfPath func(string) string
	lookup       func(string) []string
	agent        *agent.Runner
	files        *filewatch.Manager
	ws           config.WorkspaceConfig
}

func New(
	h *hub.Hub,
	host string,
	port int,
	ssePath string,
	logger *log.Logger,
	authStore *auth.Store,
	sessions *session.Manager,
	tokenTTL time.Duration,
	backendURL string,
	sourceNames func() []string,
	sourceOfPath func(string) string,
	lookup func(string) []string,
	agentRunner *agent.Runner,
	fileWatch *filewatch.Manager,
	wsCfg config.WorkspaceConfig,
) *Server {
	return &Server{
		hub:          h,
		host:         host,
		port:         port,
		path:         ssePath,
		logger:       logger,
		auth:         authStore,
		sessions:     sessions,
		tokenTTL:     tokenTTL,
		backendURL:   backendURL,
		sourceNames:  sourceNames,
		sourceOfPath: sourceOfPath,
		lookup:       lookup,
		agent:        agentRunner,
		files:        fileWatch,
		ws:           wsCfg,
	}
}

func (s *Server) Addr() string {
	return fmt.Sprintf("%s:%d", s.host, s.port)
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("POST /login", s.handleLogin)
	mux.HandleFunc("POST /sessions", s.handleCreateSession)
	mux.HandleFunc("GET /sessions", s.handleListSessions)
	mux.HandleFunc("GET /sessions/{id}", s.handleGetSession)
	mux.HandleFunc("POST /sessions/{id}/activate", s.handleActivateSession)
	mux.HandleFunc("PUT /sessions/{id}/sources", s.handleSetSessionSources)
	mux.HandleFunc("DELETE /sessions/{id}", s.handleDeleteSession)
	mux.HandleFunc("GET /models", s.handleListModels)
	mux.HandleFunc("POST /models", s.handleCreateModel)
	mux.HandleFunc("PUT /models/{id}", s.handleUpdateModel)
	mux.HandleFunc("DELETE /models/{id}", s.handleDeleteModel)
	mux.HandleFunc("POST /models/{id}/activate", s.handleActivateModel)
	mux.HandleFunc("GET /models/stages", s.handleGetModelStages)
	mux.HandleFunc("PUT /models/stages", s.handleSetModelStages)
	mux.HandleFunc("GET /sources", s.handleListSources)
	mux.HandleFunc("GET /read_cached_content", s.handleReadCachedContent)
	mux.HandleFunc("POST /sessions/{id}/invoke", s.handleInvokeAgent)
	mux.HandleFunc("GET /sessions/{id}/agent", s.handleAgentStatus)
	mux.HandleFunc("DELETE /sessions/{id}/agent", s.handleAgentStop)
	mux.HandleFunc("GET /sessions/{id}/files", s.handleListFiles)
	mux.HandleFunc("GET /sessions/{id}/monitor", s.handleSessionMonitor)
	mux.HandleFunc("GET /api/workspace/file/content", s.handleWorkspaceFileContent)
	mux.HandleFunc("POST /api/workspace/file/upload", s.handleWorkspaceFileUpload)
	mux.HandleFunc("GET /api/workspace/file/download", s.handleWorkspaceFileDownload)
	mux.HandleFunc("POST /api/workspace/directory/upload", s.handleWorkspaceDirUpload)
	mux.HandleFunc("GET /api/workspace/directory/download", s.handleWorkspaceDirDownload)
	mux.HandleFunc("GET "+s.path, s.handleSSE)
	mux.HandleFunc("/api/", s.handleAPIProxy)
	mux.HandleFunc("/", s.handleIndex)
	return s.preflight(mux)
}

// preflight intercepts CORS preflight (OPTIONS) requests before they reach the
// mux, so a single handler covers every route without conflicting with the
// method- or path-specific patterns registered above.
func (s *Server) preflight(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			s.handleOptions(w, r)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleOptions(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","subscribers":%d}`, s.hub.SubscriberCount())
}

// handleAPIProxy reverse-proxies /api/* requests to the configured ScienceFlow
// backend, so the desktop frontend can treat the gateway as the single entry
// point for all interactions (auth, state, chat, workspace, logs).
func (s *Server) handleAPIProxy(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	if s.backendURL == "" {
		http.Error(w, "backend not configured", http.StatusBadGateway)
		return
	}

	target, err := url.Parse(s.backendURL)
	if err != nil {
		http.Error(w, "invalid backend url: "+err.Error(), http.StatusBadGateway)
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	orig := proxy.Director
	proxy.Director = func(req *http.Request) {
		orig(req)
		// Preserve the incoming Authorization header (gateway token or the
		// backend's own bearer token).
		if h := r.Header.Get("Authorization"); h != "" {
			req.Header.Set("Authorization", h)
		}
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		http.Error(w, "backend unreachable: "+err.Error(), http.StatusBadGateway)
	}
	proxy.ServeHTTP(w, r)
}

// userFromRequest authenticates a control-plane request and returns the user
// name. When auth is disabled it returns ("", true).
func (s *Server) userFromRequest(w http.ResponseWriter, r *http.Request) (string, bool) {
	if s.auth == nil {
		return "", true
	}
	tok := bearerToken(r)
	if tok == "" {
		tok = r.URL.Query().Get("token")
	}
	if tok == "" {
		s.writeUnauthorized(w)
		return "", false
	}
	user, ok := s.auth.VerifyToken(tok)
	if !ok {
		s.writeUnauthorized(w)
		return "", false
	}
	return user, true
}

func bearerToken(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimSpace(strings.TrimPrefix(h, "Bearer "))
	}
	return ""
}

func (s *Server) writeUnauthorized(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusUnauthorized)
	fmt.Fprint(w, errUnauthorized)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// ---- auth: login / sessions / sources ----

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	if s.auth == nil {
		http.Error(w, "鉴权未启用", http.StatusBadRequest)
		return
	}
	var req struct {
		UserName string `json:"user_name"`
		Password string `json:"password"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if !s.auth.Verify(req.UserName, req.Password) {
		s.writeUnauthorized(w)
		return
	}
	tok, exp := s.auth.IssueToken(req.UserName, s.tokenTTL)
	writeJSON(w, http.StatusOK, map[string]any{
		"token":      tok,
		"expires_at": exp.UTC().Format(time.RFC3339),
		"user":       req.UserName,
		"sources":    s.allowedSources(req.UserName),
	})
}

// allowedSources expands a user's ACL to concrete source names (nil ACL = all).
func (s *Server) allowedSources(user string) []string {
	if s.auth == nil {
		return s.sourceNames()
	}
	list := s.auth.AllowedSources(user)
	if list == nil {
		return s.sourceNames()
	}
	known := s.sourceNames()
	knownSet := make(map[string]bool, len(known))
	for _, n := range known {
		knownSet[n] = true
	}
	var out []string
	for _, src := range list {
		if knownSet[src] {
			out = append(out, src)
		}
	}
	return out
}

// filterSources restricts requested sources to those the user may access.
func (s *Server) filterSources(user string, requested []string) []string {
	allowed := s.allowedSources(user)
	if s.auth == nil {
		return dedupe(requested)
	}
	allowedSet := make(map[string]bool, len(allowed))
	for _, n := range allowed {
		allowedSet[n] = true
	}
	var out []string
	for _, src := range requested {
		if allowedSet[src] {
			out = append(out, src)
		}
	}
	return dedupe(out)
}

func dedupe(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, v := range in {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

func (s *Server) handleCreateSession(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	var req struct {
		Sources []string `json:"sources"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	sources := s.filterSources(user, req.Sources)
	id := s.sessions.Create(user, sources)
	writeJSON(w, http.StatusOK, map[string]any{
		"session_id": id,
		"name":       s.sessions.Name(id),
		"user":       user,
		"sources":    sources,
	})
}

// sessionJSON builds the API view of one session, including a compact agent
// state so the frontend can decide what to restore when switching.
func (s *Server) sessionJSON(id string, sess *session.Session) map[string]any {
	out := map[string]any{
		"session_id":        id,
		"name":              sess.Name(),
		"user":              sess.User(),
		"sources":           sess.Sources(),
		"last_active":       sess.ChatActive().UTC().Format(time.RFC3339Nano),
		"last_active_nanos": sess.ChatActive().UnixNano(),
	}
	if s.agent != nil {
		if task := s.agent.Current(id); task != nil {
			snapshot := task.Snapshot()
			active := snapshot.Status == agent.StatusQueued || snapshot.Status == agent.StatusRunning
			status := "idle"
			if active {
				status = "active"
			}
			out["agent"] = map[string]any{"status": status, "task": snapshot}
		} else {
			out["agent"] = map[string]any{"status": "idle"}
		}
	}
	return out
}

// handleListSessions returns all live sessions owned by the caller, most
// recently active first.
func (s *Server) handleListSessions(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	sessions := s.sessions.ListByUser(user)
	items := make([]map[string]any, 0, len(sessions))
	for _, sess := range sessions {
		items = append(items, s.sessionJSON(sess.ID(), sess))
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"user":     user,
		"sessions": items,
	})
}

// handleActivateSession refreshes a session's idle TTL and returns its current
// state, so the frontend can re-attach (switch back) to a previous session.
func (s *Server) handleActivateSession(w http.ResponseWriter, r *http.Request) {
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
	sess := s.sessions.Get(id) // touches last activity
	if sess == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, s.sessionJSON(id, sess))
}

func (s *Server) handleGetSession(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	id := r.PathValue("id")
	sid := s.sessions.User(id)
	if sid == "" {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if s.auth != nil && sid != user {
		s.writeUnauthorized(w)
		return
	}
	sess := s.sessions.Get(id)
	if sess == nil {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, s.sessionJSON(id, sess))
}

func (s *Server) handleSetSessionSources(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	id := r.PathValue("id")
	if s.auth != nil && s.sessions.User(id) != user {
		s.writeUnauthorized(w)
		return
	}
	var req struct {
		Sources []string `json:"sources"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	sources := s.filterSources(user, req.Sources)
	if !s.sessions.SetSources(id, sources) {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"session_id": id,
		"sources":    sources,
	})
}

func (s *Server) handleDeleteSession(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	id := r.PathValue("id")
	if s.auth != nil && s.sessions.User(id) != user {
		s.writeUnauthorized(w)
		return
	}
	if !s.sessions.Delete(id) {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": s.agent.ModelStore().List()})
}

func (s *Server) handleCreateModel(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	var req agent.ModelCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	model, err := s.agent.ModelStore().Create(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusCreated, model)
}

func (s *Server) handleUpdateModel(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	var req agent.ModelUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	model, err := s.agent.ModelStore().Update(r.PathValue("id"), req)
	if err != nil {
		status := http.StatusBadRequest
		if err.Error() == "model not found" {
			status = http.StatusNotFound
		}
		http.Error(w, err.Error(), status)
		return
	}
	writeJSON(w, http.StatusOK, model)
}

func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	if err := s.agent.ModelStore().Delete(r.PathValue("id")); err != nil {
		status := http.StatusBadRequest
		if err.Error() == "model not found" {
			status = http.StatusNotFound
		}
		http.Error(w, err.Error(), status)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleActivateModel(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	model, err := s.agent.ModelStore().Activate(r.PathValue("id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if sessionID := strings.TrimSpace(r.URL.Query().Get("session_id")); sessionID != "" {
		if err := s.agent.ModelStore().SetSessionModel(sessionID, model.ID); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"active_model": model})
}

func (s *Server) handleGetModelStages(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	coderID, feedbackID := s.agent.ModelStore().SessionStages(strings.TrimSpace(r.URL.Query().Get("session_id")))
	writeJSON(w, http.StatusOK, map[string]any{"coder_model_id": coderID, "feedback_model_id": feedbackID})
}

func (s *Server) handleSetModelStages(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(w, r); !ok {
		return
	}
	if s.agent == nil || s.agent.ModelStore() == nil {
		http.Error(w, "agent model store not configured", http.StatusServiceUnavailable)
		return
	}
	var req struct {
		SessionID       string `json:"session_id"`
		CoderModelID    string `json:"coder_model_id"`
		FeedbackModelID string `json:"feedback_model_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if err := s.agent.ModelStore().SetSessionStages(req.SessionID, req.CoderModelID, req.FeedbackModelID); err != nil {
		status := http.StatusBadRequest
		if err.Error() == "model not found" {
			status = http.StatusNotFound
		}
		http.Error(w, err.Error(), status)
		return
	}
	coderID, feedbackID := s.agent.ModelStore().SessionStages(req.SessionID)
	writeJSON(w, http.StatusOK, map[string]any{"coder_model_id": coderID, "feedback_model_id": feedbackID})
}

func (s *Server) handleListSources(w http.ResponseWriter, r *http.Request) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}
	sources := s.allowedSources(user)
	type item struct {
		Name  string   `json:"name"`
		Files []string `json:"files"`
	}
	out := make([]item, 0, len(sources))
	for _, name := range sources {
		out = append(out, item{Name: name, Files: s.lookup(name)})
	}
	writeJSON(w, http.StatusOK, map[string]any{"sources": out})
}

// ---- streaming ----

func (s *Server) handleSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	sub := s.hub.Subscribe(256)
	defer s.hub.Unsubscribe(sub)

	var sessionID string
	var sources []string
	if s.auth != nil {
		// Session is required for an authenticated stream.
		sessionID = r.URL.Query().Get("session")
		if sessionID == "" || s.sessions.User(sessionID) == "" {
			s.writeUnauthorized(w)
			return
		}
		s.sessions.Get(sessionID) // touch activity
		sources = s.sessions.Sources(sessionID)
		sub.SetSources(sources)
	} else {
		// No auth: optional legacy ?source=a,b filter.
		sources = parseFilter(r.URL.Query().Get("source"))
		sub.SetSources(sources)
	}

	notify := s.sessions.Notify(sessionID)

	// Workspace file-tree sync: subscribe to this session's directory so the
	// frontend receives the initial tree and incremental diffs over the same
	// SSE connection. Only meaningful with an authenticated session.
	var fileCh <-chan filewatch.Event
	if s.files != nil && sessionID != "" {
		if dir, ok := s.sessionWorkspaceDir(sessionID); ok {
			var fileCancel func()
			fileCh, fileCancel = s.files.Subscribe(dir)
			defer fileCancel()
		}
	}

	fmt.Fprintf(w, ": connected\n\n")
	flusher.Flush()

	// Backfill cached log content (session switch / reconnect): stream the
	// existing content of each subscribed source's files, then fall through to
	// live tailing in the loop below.
	if sessionID != "" {
		s.writeLogBackfill(w, flusher, sessionID, sources)
	}

	ping := time.NewTicker(15 * time.Second)
	defer ping.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ping.C:
			if sessionID != "" {
				s.sessions.Get(sessionID) // keep idle session alive
			}
			fmt.Fprint(w, ": ping\n\n")
			flusher.Flush()
		case <-notify:
			sub.SetSources(s.sessions.Sources(sessionID))
		case e := <-sub.Events():
			// Per-session agent logs carry the owning session; a session-scoped
			// stream must not receive other sessions' RAW.log lines (the hub
			// only filters by source name).
			if sessionID != "" && e.Session != "" && e.Session != sessionID {
				continue
			}
			id := s.seq.Add(1)
			fmt.Fprintf(w, "id: %d\nevent: log\ndata: %s\n\n", id, e.Marshal())
			flusher.Flush()
		case fe := <-fileCh:
			data, _ := json.Marshal(fe)
			id := s.seq.Add(1)
			fmt.Fprintf(w, "id: %d\nevent: file\ndata: %s\n\n", id, data)
			flusher.Flush()
		}
	}
}

// logBackfill is the initial content snapshot of one log file, streamed over
// SSE before live tailing resumes (see writeLogBackfill).
type logBackfill struct {
	Source    string `json:"source"`
	Path      string `json:"path"`
	Size      int64  `json:"size"`
	Returned  int64  `json:"returned"`
	Truncated bool   `json:"truncated"`
	Content   string `json:"content"`
}

// sessionRawLogPath resolves the per-session agent RAW.log path, matching the
// layout the agent runner writes: <root>/task_logs/<user>/<session>/RAW.log.
func (s *Server) sessionRawLogPath(sessionID string) (string, bool) {
	if s.agent == nil {
		return "", false
	}
	user := s.sessions.User(sessionID)
	if user == "" {
		return "", false
	}
	root, err := s.agent.RootPath()
	if err != nil {
		return "", false
	}
	return filepath.ToSlash(filepath.Join(root, "task_logs", user, sessionID, "RAW.log")), true
}

// backfillFiles resolves the files whose cached content should be streamed for
// a given source, scoped to the session. Agent transcripts are per-session
// (RAW.log + the installed CLI's interaction.log); static sources map to their
// current glob matches.
func (s *Server) backfillFiles(sessionID, source string) []string {
	if source == agent.RawLogSource {
		var out []string
		if p, ok := s.sessionRawLogPath(sessionID); ok {
			out = append(out, p)
		}
		if dir, ok := s.sessionWorkspaceDir(sessionID); ok {
			if p := agent.SessionInteractionLogPath(dir); p != "" {
				if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
					out = append(out, p)
				}
			}
		}
		return out
	}
	return s.lookup(source)
}

// writeLogBackfill streams the cached (existing) content of the session's
// subscribed sources onto the SSE connection, followed by a backfill-done
// marker. The caller then enters its event loop for live tailing.
func (s *Server) writeLogBackfill(w http.ResponseWriter, flusher http.Flusher, sessionID string, sources []string) {
	seen := make(map[string]bool)
	count := 0
	for _, src := range sources {
		for _, p := range s.backfillFiles(sessionID, src) {
			if seen[p] {
				continue
			}
			seen[p] = true
			b, size, truncated, err := readSnapshot(p)
			if err != nil {
				continue
			}
			ev := logBackfill{
				Source:    src,
				Path:      p,
				Size:      size,
				Returned:  int64(len(b)),
				Truncated: truncated,
				Content:   string(b),
			}
			data, _ := json.Marshal(ev)
			id := s.seq.Add(1)
			fmt.Fprintf(w, "id: %d\nevent: backfill\ndata: %s\n\n", id, data)
			count++
		}
	}
	id := s.seq.Add(1)
	fmt.Fprintf(w, "id: %d\nevent: backfill-done\ndata: {\"count\":%d}\n\n", id, count)
	flusher.Flush()
}

func parseFilter(raw string) []string {
	if raw == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// ---- snapshot ----

type snapshot struct {
	Path      string `json:"path"`
	Size      int64  `json:"size"`
	Returned  int64  `json:"returned"`
	Truncated bool   `json:"truncated"`
	Content   string `json:"content"`
}

func (s *Server) handleReadCachedContent(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")

	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	path := q.Get("path")
	source := q.Get("source")

	var targets []string
	switch {
	case source != "":
		if s.auth != nil && !s.auth.CanAccess(user, source) {
			s.writeUnauthorized(w)
			return
		}
		targets = s.lookup(source)
		if len(targets) == 0 {
			http.Error(w, fmt.Sprintf("no files match source %q", source), http.StatusNotFound)
			return
		}
		if len(targets) > 1 {
			writeJSON(w, http.StatusOK, map[string]any{"source": source, "files": targets})
			return
		}
	case path != "":
		abs, err := filepath.Abs(path)
		if err != nil {
			http.Error(w, "invalid path: "+err.Error(), http.StatusBadRequest)
			return
		}
		abs = filepath.ToSlash(abs)
		if s.auth != nil {
			owner := s.sourceOfPath(abs)
			if owner == "" || !s.auth.CanAccess(user, owner) {
				s.writeUnauthorized(w)
				return
			}
		}
		targets = []string{abs}
	default:
		http.Error(w, "path or source query parameter is required", http.StatusBadRequest)
		return
	}

	b, size, truncated, err := readSnapshot(targets[0])
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, snapshot{
		Path:      targets[0],
		Size:      size,
		Returned:  int64(len(b)),
		Truncated: truncated,
		Content:   string(b),
	})
}

func readSnapshot(path string) ([]byte, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, false, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, 0, false, err
	}

	size := fi.Size()
	if size > maxSnapshotBytes {
		b := make([]byte, maxSnapshotBytes)
		if _, err := f.ReadAt(b, size-maxSnapshotBytes); err != nil {
			return nil, 0, false, err
		}
		return b, size, true, nil
	}
	b := make([]byte, size)
	if _, err := f.ReadAt(b, 0); err != nil {
		return nil, 0, false, err
	}
	return b, size, false, nil
}

// ---- agent invocation ----

// modePrefixRe matches the "[mode=<x>]" prefix the desktop layer injects at
// the start of the query (see doc/msg_task_type.md).
var modePrefixRe = regexp.MustCompile(`(?i)^\[mode=([a-zA-Z0-9_-]+)\]\s*`)

// parseQueryModePrefix strips a leading "[mode=<x>]" prefix from a query and
// returns the cleaned query plus the mode ("" when no prefix is present).
func parseQueryModePrefix(query string) (string, string) {
	if m := modePrefixRe.FindStringSubmatch(query); m != nil {
		return query[len(m[0]):], strings.ToLower(m[1])
	}
	return query, ""
}

func (s *Server) handleInvokeAgent(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if s.agent == nil {
		http.Error(w, "agent invocation not configured", http.StatusServiceUnavailable)
		return
	}
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
	var req struct {
		Query string `json:"query"`
		Mode  string `json:"mode"` // optional: "lite" | "heavy"
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	// The desktop layer may inject the mode as a "[mode=<x>]" prefix on the
	// query (see doc/msg_task_type.md). Accept both forms: an explicit mode
	// field wins over the prefix, the prefix wins over the default (lite).
	query, prefixMode := parseQueryModePrefix(req.Query)
	if strings.TrimSpace(query) == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	mode := req.Mode
	if mode == "" {
		mode = prefixMode
	}
	switch strings.ToLower(mode) {
	case "", agent.ModeLite:
		mode = agent.ModeLite
	case agent.ModeHeavy:
		mode = agent.ModeHeavy
	default:
		http.Error(w, fmt.Sprintf("invalid mode %q; expected lite or heavy", mode), http.StatusBadRequest)
		return
	}

	// Reject if a task for this session is still running.
	if cur := s.agent.Current(id); cur != nil && (cur.Status() == "running" || cur.Status() == "queued") {
		writeJSON(w, http.StatusConflict, map[string]any{
			"error": "an agent task is already running for this session",
			"task":  cur.Snapshot(),
		})
		return
	}

	// The agent outlives the HTTP request; detach from request cancellation.
	ctx := context.WithoutCancel(r.Context())
	task, err := s.agent.Invoke(ctx, user, id, query, mode)
	if err == nil {
		// A real chat action: stamp the display "last_active" (session list).
		s.sessions.TouchChat(id)
		// The first question names the session (shown in the session list).
		s.sessions.MaybeNameFromQuery(id, query)
	}
	if err != nil {
		if errors.Is(err, agent.ErrQueueFull) {
			queued, running := s.agent.QueueStats()
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{
				"error":   err.Error(),
				"queued":  queued,
				"running": running,
			})
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusAccepted, task.Snapshot())
}

func (s *Server) handleAgentStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if s.agent == nil {
		http.Error(w, "agent invocation not configured", http.StatusServiceUnavailable)
		return
	}
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
	task := s.agent.Current(id)
	if task == nil {
		writeJSON(w, http.StatusOK, map[string]any{"status": "idle"})
		return
	}
	snap := task.Snapshot()
	outBytes := task.Output(256 * 1024)
	queued, running := s.agent.QueueStats()
	resp := map[string]any{
		"task":    snap,
		"output":  string(outBytes),
		"running": task.Status() == "running",
		"queue_stats": map[string]int{
			"queued":  queued,
			"running": running,
		},
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleAgentStop(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if s.agent == nil {
		http.Error(w, "agent invocation not configured", http.StatusServiceUnavailable)
		return
	}
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
	task := s.agent.Current(id)
	if task == nil {
		http.Error(w, "no agent task for this session", http.StatusNotFound)
		return
	}
	if !s.agent.Stop(task.ID) {
		http.Error(w, "task not stoppable", http.StatusConflict)
		return
	}
	// Manual stop is also a user chat action for "last_active" display.
	s.sessions.TouchChat(id)
	// Give the subprocess a moment to be reaped so the response carries the
	// terminal snapshot instead of a stale "running" status.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		st := task.Status()
		if st != "running" && st != "queued" {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	writeJSON(w, http.StatusOK, task.Snapshot())
}

// ---- workspace file sync ----

// sessionWorkspaceDir resolves <root>/<user>/<session> for a session, the
// directory whose file tree is synced to the frontend.
func (s *Server) sessionWorkspaceDir(sessionID string) (string, bool) {
	if s.agent == nil {
		return "", false
	}
	user := s.sessions.User(sessionID)
	if user == "" {
		return "", false
	}
	root, err := s.agent.RootPath()
	if err != nil {
		return "", false
	}
	return filepath.Join(root, user, sessionID), true
}

// handleListFiles returns the current file tree of the session workspace.
func (s *Server) handleListFiles(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	if s.files == nil || s.agent == nil {
		http.Error(w, "file sync not configured", http.StatusServiceUnavailable)
		return
	}
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
	writeJSON(w, http.StatusOK, map[string]any{
		"root": dir,
		"tree": filewatch.Snapshot(dir),
	})
}

// ---- workspace file content ----

// maxContentBytes caps how much of a workspace file is returned; larger files
// yield only the tail with truncated=true.
const maxContentBytes = 32 * 1024 * 1024

// extContentTypes maps well-known extensions to preview content types. The
// frontend picks the renderer from content_type / encoding.
var extContentTypes = map[string]string{
	".py": "text/x-python", ".pyw": "text/x-python",
	".js": "text/javascript", ".jsx": "text/jsx",
	".ts": "text/typescript", ".tsx": "text/tsx",
	".go": "text/x-go", ".rs": "text/rust",
	".md": "text/markdown", ".markdown": "text/markdown",
	".json": "application/json", ".jsonl": "application/x-ndjson",
	".csv": "text/csv", ".tsv": "text/tab-separated-values",
	".html": "text/html", ".htm": "text/html",
	".css": "text/css", ".scss": "text/x-scss",
	".yaml": "text/yaml", ".yml": "text/yaml",
	".toml": "text/toml", ".ini": "text/plain", ".cfg": "text/plain",
	".sh": "text/x-shellscript", ".bash": "text/x-shellscript",
	".c": "text/x-c", ".h": "text/x-c", ".cpp": "text/x-c++", ".hpp": "text/x-c++",
	".java": "text/x-java", ".kt": "text/x-kotlin",
	".rb": "text/x-ruby", ".php": "text/x-php", ".swift": "text/x-swift",
	".sql": "text/x-sql", ".r": "text/x-r", ".jl": "text/x-julia",
	".xml": "text/xml", ".svg": "image/svg+xml",
	".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
	".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".ico": "image/x-icon",
	".pdf": "application/pdf",
	".txt": "text/plain", ".log": "text/plain", ".env": "text/plain",
	".pkl": "application/octet-stream", ".h5": "application/octet-stream",
	".pt": "application/octet-stream", ".pth": "application/octet-stream",
	".npy": "application/octet-stream", ".npz": "application/octet-stream",
	".zip": "application/zip", ".gz": "application/gzip", ".tar": "application/x-tar",
}

// handleWorkspaceFileContent serves one file inside a session's workspace
// directly from disk (instead of proxying to the backend). Path is relative
// to the workspace root; traversal outside the root is rejected.
func (s *Server) handleWorkspaceFileContent(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")

	user, ok := s.userFromRequest(w, r)
	if !ok {
		return
	}

	q := r.URL.Query()
	sessionID := q.Get("session_id")
	if sessionID == "" {
		http.Error(w, "session_id query parameter is required", http.StatusBadRequest)
		return
	}
	owner := s.sessions.User(sessionID)
	if owner == "" {
		http.Error(w, "session not found", http.StatusNotFound)
		return
	}
	if s.auth != nil && owner != user {
		s.writeUnauthorized(w)
		return
	}
	dir, ok := s.sessionWorkspaceDir(sessionID)
	if !ok {
		http.Error(w, "workspace root not configured", http.StatusServiceUnavailable)
		return
	}

	rel := q.Get("path")
	if rel == "" {
		http.Error(w, "path query parameter is required", http.StatusBadRequest)
		return
	}
	full, err := s.resolveWorkspacePath(dir, rel)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	f, err := os.Open(full)
	if err != nil {
		http.Error(w, "file not found: "+err.Error(), http.StatusNotFound)
		return
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if fi.IsDir() {
		http.Error(w, "path is a directory", http.StatusBadRequest)
		return
	}

	size := fi.Size()
	var b []byte
	truncated := false
	if size > maxContentBytes {
		b = make([]byte, maxContentBytes)
		if _, err := f.ReadAt(b, size-maxContentBytes); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		truncated = true
	} else {
		b = make([]byte, size)
		if _, err := f.ReadAt(b, 0); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	contentType := extContentTypes[strings.ToLower(filepath.Ext(full))]
	if contentType == "" {
		contentType = http.DetectContentType(b)
	}

	encoding := "utf8"
	content := string(b)
	if contentType == "application/octet-stream" ||
		strings.HasPrefix(contentType, "image/") ||
		strings.HasPrefix(contentType, "application/pdf") ||
		strings.HasPrefix(contentType, "application/zip") ||
		strings.HasPrefix(contentType, "application/gzip") ||
		strings.HasPrefix(contentType, "application/x-tar") {
		encoding = "base64"
		content = base64.StdEncoding.EncodeToString(b)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"path":         rel,
		"size":         size,
		"returned":     int64(len(b)),
		"truncated":    truncated,
		"mtime":        fi.ModTime().UnixNano(),
		"content_type": contentType,
		"encoding":     encoding,
		"content":      content,
	})
}

// resolveWorkspacePath joins a relative path onto the workspace root and
// verifies the result stays inside it (blocks ".." and absolute escapes).
func (s *Server) resolveWorkspacePath(root, rel string) (string, error) {
	clean := filepath.Clean(filepath.ToSlash(rel))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || filepath.IsAbs(clean) {
		return "", fmt.Errorf("invalid path %q", rel)
	}
	full := filepath.Join(root, filepath.FromSlash(clean))
	fullAbs := filepath.ToSlash(filepath.Clean(full))
	rootAbs := filepath.ToSlash(filepath.Clean(root))
	if fullAbs != rootAbs && !strings.HasPrefix(fullAbs, rootAbs+"/") {
		return "", fmt.Errorf("path %q escapes workspace root", rel)
	}
	return full, nil
}

func (s *Server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, indexHTML)
}

const indexHTML = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Log Gateway SSE</title>
<style>
body{font-family:monospace;margin:16px;background:#111;color:#90ee90}
pre{background:#000;padding:12px;height:80vh;overflow:auto;border:1px solid #333}
input{padding:4px;margin:2px}
</style>
</head>
<body>
<h3>Log Gateway &mdash; live tail</h3>
<input id="u" placeholder="user_name" value="admin">
<input id="p" type="password" placeholder="password" value="admin123">
<button onclick="connect()">connect</button>
<pre id="log"></pre>
<script>
var es = null;
function append(e) {
  var d = JSON.parse(e.data);
  var pre = document.getElementById('log');
  var body = d.message;
  if (d.encoding === 'base64') { body = '[base64] ' + d.data; }
  pre.textContent += '[' + d.timestamp + '] ' + d.file + ' (' + d.mode + '): ' + body + '\n';
  pre.scrollTop = pre.scrollHeight;
}
async function connect() {
  if (es) es.close();
  var u = document.getElementById('u').value;
  var p = document.getElementById('p').value;
  var url = '/events';
  try {
    var lr = await fetch('/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_name:u, password:p})});
    if (lr.ok) {
      var l = await lr.json();
      var sr = await fetch('/sessions', {method:'POST', headers:{'Content-Type':'application/json','Authorization':'Bearer '+l.token}, body: JSON.stringify({sources: l.sources})});
      if (sr.ok) {
        var s = await sr.json();
        url = '/events?session=' + s.session_id;
      }
    }
  } catch (err) {}
  es = new EventSource(url);
  es.addEventListener('log', append);
}
</script>
</body>
</html>
`
