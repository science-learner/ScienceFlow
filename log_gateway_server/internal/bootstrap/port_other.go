//go:build !linux && !darwin && !windows

package bootstrap

import "fmt"

// lookupPortOwner is unsupported on this platform; the caller falls back to
// the "occupied but owner unknown" path.
func lookupPortOwner(port int) (*PortOwner, error) {
	return nil, fmt.Errorf("port owner lookup is not supported on this platform")
}

// processCommand is unavailable on this platform.
func processCommand(pid int) string { return "" }
