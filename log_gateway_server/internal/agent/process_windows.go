//go:build windows

package agent

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// terminateProcess stops the agent subprocess and its children on Windows.
// Console applications have no SIGTERM equivalent, so taskkill /T /F is used
// (terminating the whole tree).
func terminateProcess(cmd *exec.Cmd) error {
	out, err := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("taskkill pid %d: %w (%s)", cmd.Process.Pid, err, strings.TrimSpace(string(out)))
	}
	return nil
}
