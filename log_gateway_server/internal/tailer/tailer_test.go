package tailer

import "testing"

func TestCleanInteractionLine(t *testing.T) {
	cases := []struct{ in, want string }{
		{
			in:   "2026-09-23 08:34:03 | INFO     | [draft|?] \x1b[96m[user] hello\x1b[0m",
			want: "[draft|?] [user] hello",
		},
		{
			// heavy worker transcripts add a loguru logger-name segment.
			in:   "2026-10-15 03:14:47 | INFO     | scienceflow | \x1b[96m[user] You are solving one ML task.\x1b[0m",
			want: "[user] You are solving one ML task.",
		},
		{
			in:   "2026-09-23 08:34:12 | INFO     | [draft|?] \x1b[32m[tool-call] ls {\"path\": \".\"}\x1b[0m",
			want: "[draft|?] [tool-call] ls {\"path\": \".\"}",
		},
		{
			// continuation lines (tool output) have no prefix: pass through,
			// only ANSI escapes are removed.
			in:   "\x1b[2mDIR logs/\x1b[0m",
			want: "DIR logs/",
		},
	}
	for _, c := range cases {
		if got := cleanInteractionLine(c.in); got != c.want {
			t.Errorf("cleanInteractionLine(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSessionFromAgentLogPath(t *testing.T) {
	cases := []struct {
		path string
		want string
	}{
		{"/ws/task_logs/admin/s1/RAW.log", "s1"},
		{"/ws/admin/s1/run/.logs/interaction.log", "s1"},
		{"/ws/admin/s1/run/logs/interaction.log", ""},
		{"/ws/admin/s1/run/task_logs/parallel_subprocess.log", "s1"},
		{"/ws/admin/s1/task_logs/parallel_subprocess.log", ""},
		{"/logs/sample.log", ""},
	}
	for _, c := range cases {
		if got := sessionFromAgentLogPath(c.path); got != c.want {
			t.Errorf("sessionFromAgentLogPath(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}
