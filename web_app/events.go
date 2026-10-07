package main

import "github.com/wailsapp/wails/v3/pkg/application"

// emitEvent forwards an event to the frontend through the Wails v3 event
// manager, replacing the v2 runtime.EventsEmit bridge.
func emitEvent(name string, data any) {
	if app := application.Get(); app != nil {
		app.Event.Emit(name, data)
	}
}
