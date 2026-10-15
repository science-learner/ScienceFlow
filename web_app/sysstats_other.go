//go:build !windows && !linux

package main

// platformSystemStats returns zeroed values on platforms without a native
// collector (macOS server-mode deployments). Windows and Linux report real
// counters via sysstats_windows.go / sysstats_linux.go.
func platformSystemStats() (SystemStats, error) {
	return SystemStats{}, nil
}
