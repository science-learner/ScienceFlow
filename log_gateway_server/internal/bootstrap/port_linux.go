//go:build linux

package bootstrap

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// lookupPortOwner resolves the process listening on port using ss (iproute2),
// lsof, or fuser, in that order. A nil owner without error means the port is
// occupied but no owner is visible (e.g. missing permissions); a non-nil
// error means none of the tools could be queried.
func lookupPortOwner(port int) (*PortOwner, error) {
	var lastErr error

	if path, err := exec.LookPath("ss"); err == nil {
		out, _, e := runCapture([]string{path},
			[]string{"-ltnpH", fmt.Sprintf("sport = :%d", port)}, 10*time.Second)
		if e == nil {
			if owners := parseSSOwners(out, port); len(owners) > 0 {
				return finalizeOwner(&owners[0]), nil
			}
		} else {
			lastErr = fmt.Errorf("ss -ltnp: %v", e)
		}
	} else {
		lastErr = fmt.Errorf("ss not found: %v", err)
	}

	if path, err := exec.LookPath("lsof"); err == nil {
		// lsof exits non-zero when it finds nothing, so parse stdout anyway.
		out, _, _ := runCapture([]string{path}, lsofArgs(port), 10*time.Second)
		if owners := parseLsofOwners(out); len(owners) > 0 {
			return finalizeOwner(&owners[0]), nil
		}
	} else {
		lastErr = fmt.Errorf("lsof not found: %v", err)
	}

	if path, err := exec.LookPath("fuser"); err == nil {
		// Plain (non -v) fuser prints bare pids on stdout; ignore its exit
		// code (non-zero when no process matches).
		out, _, _ := runCapture([]string{path}, []string{"-n", "tcp", strconv.Itoa(port)}, 10*time.Second)
		if owners := parseFuserOwners(out); len(owners) > 0 {
			return finalizeOwner(&owners[0]), nil
		}
	} else {
		lastErr = fmt.Errorf("fuser not found: %v", err)
	}

	return nil, lastErr
}

// processCommand returns the full command line of pid via /proc, or "" when
// unavailable (e.g. other user's process).
func processCommand(pid int) string {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return ""
	}
	parts := strings.Split(strings.TrimRight(string(b), "\x00"), "\x00")
	return strings.TrimSpace(strings.Join(parts, " "))
}
