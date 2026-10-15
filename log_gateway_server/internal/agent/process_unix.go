//go:build !windows

package agent

import (
	"os/exec"
	"syscall"
)

// terminateProcess asks the agent subprocess to exit gracefully (SIGTERM).
// The installed scienceflow CLI traps SIGTERM, sweeps its own child process
// groups (heavy-mode LNR workers from `cli parallel`), and then exits — so a
// plain SIGTERM is enough to stop the whole tree. SIGKILL is deliberately not
// used: it would bypass that cleanup and orphan the research workers.
func terminateProcess(cmd *exec.Cmd) error {
	return syscall.Kill(cmd.Process.Pid, syscall.SIGTERM)
}
