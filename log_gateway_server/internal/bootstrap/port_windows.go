//go:build windows

package bootstrap

import (
	"fmt"
	"time"
)

// lookupPortOwner resolves the process listening on port using netstat and
// tasklist. A nil owner without error means the port is occupied but no owner
// is visible; a non-nil error means the tools could not be queried.
func lookupPortOwner(port int) (*PortOwner, error) {
	out, _, err := runCapture([]string{"netstat"}, []string{"-ano", "-p", "tcp"}, 10*time.Second)
	if err != nil {
		return nil, fmt.Errorf("netstat: %v", err)
	}
	owners := parseNetstatOwners(out, port)
	if len(owners) == 0 {
		return nil, nil
	}
	o := owners[0]
	o.Name = windowsImageName(o.PID)
	return &o, nil
}

// windowsImageName resolves a pid's image name via tasklist.
func windowsImageName(pid int) string {
	out, _, err := runCapture([]string{"tasklist"},
		[]string{"/FI", fmt.Sprintf("PID eq %d", pid), "/FO", "CSV", "/NH"}, 10*time.Second)
	if err != nil {
		return ""
	}
	name := parseTasklistImage(out)
	if name == "" {
		return "?"
	}
	return name
}

// processCommand is unavailable on Windows.
func processCommand(pid int) string { return "" }
