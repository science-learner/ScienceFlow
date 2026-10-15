package bootstrap

import (
	"bufio"
	"bytes"
	"net"
	"strings"
	"testing"
)

func TestParseSSOwners(t *testing.T) {
	out := "" +
		"LISTEN 0      4096         0.0.0.0:8901       0.0.0.0:*    users:((\"python3\",pid=4567,fd=3))\n" +
		"LISTEN 0      4096            [::]:8901          [::]:*    users:((\"python3\",pid=4567,fd=4))\n" +
		"LISTEN 0      4096         0.0.0.0:22         0.0.0.0:*    users:((\"sshd\",pid=777,fd=3))\n" +
		"LISTEN 0      4096         0.0.0.0:8901       0.0.0.0:*\n" // no permission: no users field
	owners := parseSSOwners(out, 8901)
	if len(owners) != 1 {
		t.Fatalf("owners = %v, want 1 (deduped, port-filtered)", owners)
	}
	if owners[0].PID != 4567 || owners[0].Name != "python3" {
		t.Errorf("owner = %+v, want python3/4567", owners[0])
	}
	// pid-only form (no name) still yields the pid.
	out2 := "LISTEN 0 4096 0.0.0.0:8901 0.0.0.0:* users:(pid=99,fd=3))\n"
	owners2 := parseSSOwners(out2, 8901)
	if len(owners2) != 1 || owners2[0].PID != 99 {
		t.Errorf("pid-only owners = %+v, want 1 entry with pid 99", owners2)
	}
}

func TestParseLsofOwners(t *testing.T) {
	out := "" +
		"COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n" +
		"lgw     12345 root    7u  IPv4 123456      0t0  TCP *:8901 (LISTEN)\n" +
		"python3 23456 root    4u  IPv6 234567      0t0  TCP *:8901 (LISTEN)\n"
	owners := parseLsofOwners(out)
	if len(owners) != 2 {
		t.Fatalf("owners = %v, want 2", owners)
	}
	if owners[0].PID != 12345 || owners[0].Name != "lgw" {
		t.Errorf("owner[0] = %+v", owners[0])
	}
	if got := parseLsofOwners("COMMAND   PID USER"); len(got) != 0 {
		t.Errorf("header-only owners = %v, want none", got)
	}
}

func TestParseNetstatOwners(t *testing.T) {
	out := "" +
		"  TCP    0.0.0.0:135      0.0.0.0:0       LISTENING       1234\n" +
		"  TCP    0.0.0.0:8901     0.0.0.0:0       LISTENING       4567\n" +
		"  TCP    [::]:8901        [::]:0          LISTENING       4567\n" +
		"  TCP    127.0.0.1:8901   127.0.0.1:55000 ESTABLISHED     9999\n" +
		"  TCP    0.0.0.0:8080     0.0.0.0:0       LISTENING       7777\n"
	owners := parseNetstatOwners(out, 8901)
	if len(owners) != 1 || owners[0].PID != 4567 {
		t.Errorf("owners = %+v, want pid 4567 only", owners)
	}
}

func TestParseFuserOwners(t *testing.T) {
	owners := parseFuserOwners("  5816 991 \n")
	if len(owners) != 2 || owners[0].PID != 5816 || owners[1].PID != 991 {
		t.Errorf("owners = %+v, want pids 5816, 991", owners)
	}
	if got := parseFuserOwners("   "); len(got) != 0 {
		t.Errorf("blank owners = %v, want none", got)
	}
}

func TestParseTasklistImage(t *testing.T) {
	out := "\"lgw.exe\",\"4567\",\"Console\",\"1\",\"12,345 K\"\n"
	if got := parseTasklistImage(out); got != "lgw.exe" {
		t.Errorf("parseTasklistImage = %q, want lgw.exe", got)
	}
	if got := parseTasklistImage(""); got != "" {
		t.Errorf("empty parseTasklistImage = %q", got)
	}
}

func TestAskPort(t *testing.T) {
	var buf bytes.Buffer
	u := &ui{w: &buf, color: false}

	// invalid, invalid, valid -> valid port returned.
	in := bufio.NewReader(strings.NewReader("abc\n0\n39123\n"))
	if got := askPort(u, in, 8901); got != 39123 {
		t.Errorf("askPort = %d, want 39123", got)
	}
	// empty line -> default.
	in = bufio.NewReader(strings.NewReader("\n"))
	if got := askPort(u, in, 8901); got != 8901 {
		t.Errorf("askPort empty = %d, want 8901", got)
	}
	// EOF -> default.
	in = bufio.NewReader(strings.NewReader(""))
	if got := askPort(u, in, 8901); got != 8901 {
		t.Errorf("askPort EOF = %d, want 8901", got)
	}
	// out of range then in range.
	in = bufio.NewReader(strings.NewReader("70000\n65535\n"))
	if got := askPort(u, in, 8901); got != 65535 {
		t.Errorf("askPort range = %d, want 65535", got)
	}
}

func TestPortAvailable(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	if portAvailable("127.0.0.1", port) {
		t.Errorf("port %d should be reported occupied", port)
	}
	_ = ln.Close()
	if !portAvailable("127.0.0.1", port) {
		t.Errorf("port %d should be reported free after close", port)
	}
}

func TestListenAddr(t *testing.T) {
	if got := listenAddr("", 80); got != "0.0.0.0:80" {
		t.Errorf("listenAddr empty host = %q", got)
	}
	if got := listenAddr("127.0.0.1", 8901); got != "127.0.0.1:8901" {
		t.Errorf("listenAddr = %q", got)
	}
}

func TestClip(t *testing.T) {
	if got := clip("short", 10); got != "short" {
		t.Errorf("clip short = %q", got)
	}
	if got := clip("abcdefghij", 5); got != "abcd…" {
		t.Errorf("clip long = %q", got)
	}
	if got := clip("启动引导测试", 3); got != "启动…" {
		t.Errorf("clip CJK = %q", got)
	}
}
