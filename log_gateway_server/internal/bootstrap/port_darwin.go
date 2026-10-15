//go:build darwin

package bootstrap

import (
	"fmt"
	"os/exec"
	"time"
)

// lookupPortOwner resolves the process listening on port using lsof (macOS
// has no ss). A nil owner without error means the port is occupied but no
// owner is visible; a non-nil error means lsof could not be queried.
func lookupPortOwner(port int) (*PortOwner, error) {
	path, err := exec.LookPath("lsof")
	if err != nil {
		return nil, fmt.Errorf("lsof not found: %v", err)
	}
	// lsof exits non-zero when it finds nothing, so parse stdout anyway.
	out, _, _ := runCapture([]string{path}, lsofArgs(port), 10*time.Second)
	if owners := parseLsofOwners(out); len(owners) > 0 {
		o := owners[0]
		return &o, nil
	}
	return nil, nil
}

// processCommand is unavailable on macOS.
func processCommand(pid int) string { return "" }
