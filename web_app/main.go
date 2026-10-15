package main

import (
	"embed"
	"log"

	"github.com/wailsapp/wails/v3/pkg/application"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	app := NewApp()

	// Server mode (-tags server): host/port come from app_config.yaml and are
	// also overridable with WAILS_SERVER_HOST / WAILS_SERVER_PORT. The desktop
	// build ignores these options.
	serverHost, serverPort := resolveServerAddress()
	log.Printf("ScienceFlow (%s): gateway=%s server(config)=%s:%d", buildMode, app.GetGatewayURL(), serverHost, serverPort)

	wailsApp := application.New(application.Options{
		Name: "ScienceFlow",
		Services: []application.Service{
			application.NewService(app),
		},
		Assets: application.AssetOptions{
			Handler: application.AssetFileServerFS(assets),
		},
		Server: application.ServerOptions{
			Host: serverHost,
			Port: serverPort,
		},
		Mac: application.MacOptions{
			ApplicationShouldTerminateAfterLastWindowClosed: true,
		},
	})

	wailsApp.Window.NewWithOptions(application.WebviewWindowOptions{
		Title:            "ScienceFlow",
		Width:            1440,
		Height:           900,
		MinWidth:         960,
		MinHeight:        600,
		BackgroundColour: application.NewRGB(11, 13, 16),
		URL:              "/",
	})

	if err := wailsApp.Run(); err != nil {
		log.Fatal(err)
	}
}
