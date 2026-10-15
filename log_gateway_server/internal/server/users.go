package server

import (
	"encoding/json"
	"errors"
	"net/http"

	"log_gateway_server/internal/auth"
)

// ---- super-admin account management (/users) ----
//
// All endpoints require a Bearer token whose account carries role: admin in
// AUTH.yaml. Mutations are persisted atomically to AUTH.yaml and applied to
// the live store immediately (hot update, no restart). Deleting an account or
// changing its password also invalidates that account's outstanding tokens.

// requireAdmin authenticates the caller and enforces the admin role.
func (s *Server) requireAdmin(w http.ResponseWriter, r *http.Request) (string, bool) {
	if s.auth == nil {
		http.Error(w, "鉴权未启用，用户管理不可用", http.StatusForbidden)
		return "", false
	}
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return "", false
	}
	if !s.auth.IsAdmin(user) {
		http.Error(w, "需要超级管理员权限", http.StatusForbidden)
		return "", false
	}
	return user, true
}

// userWriteRequest is the shared body for create/update. Pointer fields keep
// "absent" distinguishable from "set to empty" during updates.
type userWriteRequest struct {
	Password *string   `json:"password"`
	Sources  *[]string `json:"sources"`
	Role     *string   `json:"role"`
}

func decodeUserWrite(w http.ResponseWriter, r *http.Request, req any) bool {
	if err := json.NewDecoder(r.Body).Decode(req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return false
	}
	return true
}

// authErrorStatus maps account-management errors to HTTP status codes.
func authErrorStatus(err error) int {
	switch {
	case errors.Is(err, auth.ErrUserNotFound):
		return http.StatusNotFound
	case errors.Is(err, auth.ErrUserExists), errors.Is(err, auth.ErrLastAdmin):
		return http.StatusConflict
	default:
		return http.StatusBadRequest
	}
}

// userView returns the API projection of one account.
func (s *Server) userView(name string) (auth.UserView, bool) {
	for _, v := range s.auth.ListUsers() {
		if v.Name == name {
			return v, true
		}
	}
	return auth.UserView{}, false
}

// handleListUsers returns every account (never passwords).
func (s *Server) handleListUsers(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.requireAdmin(w, r); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"users": s.auth.ListUsers()})
}

// handleCreateUser adds an account: {"name","password","sources"?,"role"?}.
func (s *Server) handleCreateUser(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.requireAdmin(w, r); !ok {
		return
	}
	var req struct {
		Name string `json:"name"`
		userWriteRequest
	}
	if !decodeUserWrite(w, r, &req) {
		return
	}
	if req.Password == nil {
		http.Error(w, "password is required", http.StatusBadRequest)
		return
	}
	sources := []string(nil)
	if req.Sources != nil {
		sources = *req.Sources
	}
	role := ""
	if req.Role != nil {
		role = *req.Role
	}
	if err := s.auth.CreateUser(req.Name, *req.Password, sources, role); err != nil {
		http.Error(w, err.Error(), authErrorStatus(err))
		return
	}
	view, _ := s.userView(req.Name)
	writeJSON(w, http.StatusCreated, map[string]any{"user": view})
}

// handleUpdateUser changes an account: any of {"password","sources","role"}.
func (s *Server) handleUpdateUser(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.requireAdmin(w, r); !ok {
		return
	}
	var req userWriteRequest
	if !decodeUserWrite(w, r, &req) {
		return
	}
	if req.Password == nil && req.Sources == nil && req.Role == nil {
		http.Error(w, "至少需要提供 password、sources 或 role 之一", http.StatusBadRequest)
		return
	}
	name := r.PathValue("name")
	patch := auth.UserPatch{Password: req.Password, Sources: req.Sources, Role: req.Role}
	if err := s.auth.UpdateUser(name, patch); err != nil {
		http.Error(w, err.Error(), authErrorStatus(err))
		return
	}
	view, _ := s.userView(name)
	writeJSON(w, http.StatusOK, map[string]any{"user": view})
}

// handleDeleteUser removes an account.
func (s *Server) handleDeleteUser(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.requireAdmin(w, r); !ok {
		return
	}
	if err := s.auth.DeleteUser(r.PathValue("name")); err != nil {
		http.Error(w, err.Error(), authErrorStatus(err))
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
