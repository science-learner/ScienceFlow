package main

import (
	"context"

	"github.com/wailsapp/wails/v3/pkg/application"
)

// App holds the desktop application state: the shared context and the
// configured gateway base URL. All agent interaction (invoke + transcript)
// flows through the log gateway; there is no backend chat bridge anymore.
type App struct {
	ctx        context.Context
	gatewayURL string
	gwStream   *gatewayStreamManager
}

// NewApp creates the application struct. The gateway base URL is resolved
// from app_config.yaml (next to the executable, then the working directory),
// falling back to the built-in default.
func NewApp() *App {
	return &App{
		gatewayURL: resolveGatewayURL(),
	}
}

// ServiceStartup implements the Wails v3 service lifecycle. It saves the
// context so we can call the runtime and initialises the SSE bridges.
func (a *App) ServiceStartup(ctx context.Context, options application.ServiceOptions) error {
	a.ctx = ctx
	a.gwStream = newGatewayStreamManager(ctx)
	return nil
}

// ServiceShutdown stops the SSE bridges when the app quits.
func (a *App) ServiceShutdown() error {
	a.shutdown()
	return nil
}

func (a *App) shutdown() {
	if a.gwStream != nil {
		a.gwStream.Stop()
	}
}

// SetGatewayURL changes the log gateway base URL at runtime.
func (a *App) SetGatewayURL(url string) {
	a.gatewayURL = normalizeBaseURL(url, defaultGatewayURL)
}

// GetGatewayURL returns the currently configured log gateway base URL.
func (a *App) GetGatewayURL() string {
	return a.gatewayURL
}

// GatewayStartStream (re)starts the log gateway SSE stream for a session.
func (a *App) GatewayStartStream(sessionID string) {
	if a.gwStream == nil {
		return
	}
	a.gwStream.Start(a.gatewayURL, sessionID)
}

// GatewayStopStream stops the current log gateway SSE stream.
func (a *App) GatewayStopStream() {
	if a.gwStream != nil {
		a.gwStream.Stop()
	}
}