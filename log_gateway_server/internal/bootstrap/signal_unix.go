//go:build !windows

package bootstrap

import (
	"fmt"
	"syscall"
)

// terminateProcess asks the process to exit gracefully (SIGTERM), letting an
// old gateway instance flush its registry checkpoint.
func terminateProcess(pid int) error {
	return syscall.Kill(pid, syscall.SIGTERM)
}

// forceKillProcess terminates the process unconditionally (SIGKILL).
func forceKillProcess(pid int) error {
	return syscall.Kill(pid, syscall.SIGKILL)
}

// killHint returns the manual command shown when automatic termination is
// not possible.
func killHint(pid int) string {
	return fmt.Sprintf("kill %d", pid)
}
