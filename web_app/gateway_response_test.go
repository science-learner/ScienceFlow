package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newHTMLServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, `<!doctype html><html><body><div id="app"></div></body></html>`)
	}))
}

// A success (200) HTML response means the configured address is not the log
// gateway (SPA fallback, reverse-proxy error page, or the web UI itself).
// Every typed call must fail with an actionable error naming the endpoint and
// the address instead of a raw "invalid character '<'" JSON error.
func TestGatewayHTMLResponseDetection(t *testing.T) {
	srv := newHTMLServer(t)
	defer srv.Close()
	a := &App{gatewayURL: srv.URL}

	if _, err := a.GatewayLogin(GatewayLoginRequest{UserName: "u", Password: "p"}); err == nil {
		t.Fatal("GatewayLogin: expected an error for an HTML response")
	} else if !strings.Contains(err.Error(), "HTML") || !strings.Contains(err.Error(), srv.URL) || !strings.Contains(err.Error(), "/login") {
		t.Fatalf("GatewayLogin error is not actionable: %v", err)
	}

	if _, err := a.GatewayFetchFiles("token", "session-1"); err == nil {
		t.Fatal("GatewayFetchFiles: expected an error for an HTML response")
	} else if !strings.Contains(err.Error(), "HTML") || !strings.Contains(err.Error(), "/files") {
		t.Fatalf("GatewayFetchFiles error is not actionable: %v", err)
	}

	if _, err := a.GatewayFetchMonitor("token", "session-1"); err == nil {
		t.Fatal("GatewayFetchMonitor: expected an error for an HTML response")
	} else if !strings.Contains(err.Error(), "HTML") || !strings.Contains(err.Error(), "/monitor") {
		t.Fatalf("GatewayFetchMonitor error is not actionable: %v", err)
	}
}

// Valid JSON responses (and empty 204 bodies) must keep working.
func TestGatewayJSONResponsesStillWork(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/login":
			fmt.Fprint(w, `{"token":"t","user":"u","sources":["agent-logs"]}`)
		case "/sessions/session-1/files":
			fmt.Fprint(w, `{"root":"/tmp/ws","tree":[{"name":"a.md","path":"a.md","type":"file","size":1,"mtime":2}]}`)
		case "/sessions/session-1/monitor":
			fmt.Fprint(w, `{"session_id":"session-1","resources":{"cpu_percent":1}}`)
		case "/sessions/session-1":
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := &App{gatewayURL: srv.URL}

	login, err := a.GatewayLogin(GatewayLoginRequest{UserName: "u", Password: "p"})
	if err != nil || login.Token != "t" {
		t.Fatalf("GatewayLogin: err=%v result=%+v", err, login)
	}

	files, err := a.GatewayFetchFiles("t", "session-1")
	if err != nil || len(files.Tree) != 1 || files.Tree[0].Path != "a.md" {
		t.Fatalf("GatewayFetchFiles: err=%v result=%+v", err, files)
	}

	mon, err := a.GatewayFetchMonitor("t", "session-1")
	if err != nil || mon.SessionID != "session-1" {
		t.Fatalf("GatewayFetchMonitor: err=%v result=%+v", err, mon)
	}

	if err := a.GatewayDeleteSession("t", "session-1"); err != nil {
		t.Fatalf("GatewayDeleteSession (204): %v", err)
	}
}
