//go:build windows

package bootstrap

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// terminateProcess stops the process via taskkill (Windows has no SIGTERM).
func terminateProcess(pid int) error {
	return runTaskkill(pid, false)
}

// forceKillProcess terminates the process forcefully via taskkill /F.
func forceKillProcess(pid int) error {
	return runTaskkill(pid, true)
}

func runTaskkill(pid int, force bool) error {
	args := []string{"/PID", strconv.Itoa(pid)}
	if force {
		args = append(args, "/F")
	}
	out, err := exec.Command("taskkill", args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("taskkill: %v (%s)", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// killHint returns the manual command shown when automatic termination is
// not possible.
func killHint(pid int) string {
	return fmt.Sprintf("taskkill /F /PID %d", pid)
}
