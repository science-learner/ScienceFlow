package bootstrap

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestCmpVer(t *testing.T) {
	cases := []struct {
		a, b string
		want int // -1: a<b, 0: a==b, +1: a>b; -2: uncomparable
	}{
		{"0.2.0b6", "0.2.0b7", -1},
		{"0.2.0b7", "0.2.0b6", 1},
		{"0.2.0b5", "0.2.0b5", 0},
		{"0.2.0b6", "0.2.0", -1},        // pre-release sorts before final
		{"0.2.0", "0.2.0b5", 1},         // final after pre-release
		{"1.0", "1.0.0", 0},             // zero padding
		{"0.10.0", "0.9.9", 1},          // numeric, not lexicographic
		{"0.2.0rc1", "0.2.0b7", 1},      // rc after beta
		{"0.2.0b10", "0.2.0b9", 1},      // pre-release number
		{"0.2.0a1", "0.2.0.dev1", 1},    // dev before alpha
		{"0.2.0.post1", "0.2.0", 1},     // post after final
		{"1.2.3", "1.2.4b1", -1},        // final < next pre-release
		{"v0.2.0b6", "0.2.0b6", 0},      // leading v
		{"2026.09.01", "2026.08.31", 1}, // date-like versions
		{"not-a-version", "0.1.0", -2},
		{"0.1.0", "also-bad", -2},
	}
	for _, c := range cases {
		got, ok := cmpVer(c.a, c.b)
		if c.want == -2 {
			if ok {
				t.Errorf("cmpVer(%q, %q): expected uncomparable, got %d", c.a, c.b, got)
			}
			continue
		}
		if !ok {
			t.Errorf("cmpVer(%q, %q): unexpectedly uncomparable", c.a, c.b)
			continue
		}
		if got != c.want {
			t.Errorf("cmpVer(%q, %q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestParsePipIndexVersions(t *testing.T) {
	out := "scienceflow (0.2.0b7)\nAvailable versions: 0.2.0b7, 0.2.0b6, 0.2.0b5, 0.2.0b4, 0.2.0b3\n  INSTALLED: 0.2.0b6\n"
	if got := parsePipIndexVersions(out, "scienceflow"); got != "0.2.0b7" {
		t.Errorf("primary form: got %q", got)
	}
	// Only the list form (older pip builds).
	out2 := "Available versions: 1.4.2, 1.4.1, 1.3.0\n"
	if got := parsePipIndexVersions(out2, "six"); got != "1.4.2" {
		t.Errorf("fallback form: got %q", got)
	}
	if got := parsePipIndexVersions("no match here", "scienceflow"); got != "" {
		t.Errorf("no-match: got %q", got)
	}
	// The package line is matched case-insensitively by name.
	out3 := "ScienceFlow (0.3.0)\n"
	if got := parsePipIndexVersions(out3, "scienceflow"); got != "0.3.0" {
		t.Errorf("case-insensitive: got %q", got)
	}
}

func TestSplitCommand(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"python", []string{"python"}},
		{"uv run python", []string{"uv", "run", "python"}},
		{`"/path/with space/python" -u`, []string{"/path/with space/python", "-u"}},
		{"  python3   -u  ", []string{"python3", "-u"}},
		{"", nil},
	}
	for _, c := range cases {
		got := splitCommand(c.in)
		if len(got) != len(c.want) {
			t.Errorf("splitCommand(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("splitCommand(%q)[%d] = %q, want %q", c.in, i, got[i], c.want[i])
			}
		}
	}
}

func TestPrepareBase(t *testing.T) {
	cases := []struct {
		in, want []string
	}{
		{[]string{"python"}, []string{"python"}},
		{[]string{"python3", "-u"}, []string{"python3", "-u"}},
		{[]string{"uv", "run", "python"}, []string{"uv", "run", "--no-sync", "python"}},
		{[]string{"uv", "run", "--no-sync", "python"}, []string{"uv", "run", "--no-sync", "python"}},
	}
	for _, c := range cases {
		got := prepareBase(c.in)
		if strings.Join(got, " ") != strings.Join(c.want, " ") {
			t.Errorf("prepareBase(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestPipArgs(t *testing.T) {
	base := []string{"python"}
	got := strings.Join(pipArgs(base, "scienceflow", true, true), " ")
	want := "python -m pip install --upgrade --pre scienceflow"
	if got != want {
		t.Errorf("pipArgs upgrade+pre = %q, want %q", got, want)
	}
	got = strings.Join(pipArgs(base, "scienceflow", false, false), " ")
	want = "python -m pip install scienceflow"
	if got != want {
		t.Errorf("pipArgs plain = %q, want %q", got, want)
	}
}

func TestDistOf(t *testing.T) {
	cases := map[string]string{
		"scienceflow.cli": "scienceflow",
		"scienceflow":     "scienceflow",
		"a.b.c":           "a",
		".weird":          ".weird",
	}
	for in, want := range cases {
		if got := distOf(in); got != want {
			t.Errorf("distOf(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestDispWidth(t *testing.T) {
	cases := map[string]int{
		"abc":         3,
		"启动引导":        8,
		"Python 3.12": 11,
		"网关配置 panel":  14,
	}
	for in, want := range cases {
		if got := dispWidth(in); got != want {
			t.Errorf("dispWidth(%q) = %d, want %d", in, got, want)
		}
	}
	if got := dispWidth(stripANSI("\x1b[1mbold\x1b[0m")); got != 4 {
		t.Errorf("stripANSI width = %d, want 4", got)
	}
}

func TestFormatDur(t *testing.T) {
	cases := map[time.Duration]string{
		24 * time.Hour:              "24h",
		90 * time.Minute:            "1h30m",
		45 * time.Second:            "45s",
		0:                           "0s",
		-1 * time.Second:            "0s",
		3*time.Hour + 5*time.Second: "3h5s",
	}
	for in, want := range cases {
		if got := formatDur(in); got != want {
			t.Errorf("formatDur(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestIndentWriter(t *testing.T) {
	var buf bytes.Buffer
	iw := &indentWriter{w: &buf, prefix: "| ", atStart: true}
	// Two writes + two "processes" writing interleaved-ish chunks.
	_, _ = iw.Write([]byte("line one\nline "))
	_, _ = iw.Write([]byte("two\nno newline tail"))
	_, _ = iw.Write([]byte("\nfinal\n"))
	want := "| line one\n| line two\n| no newline tail\n| final\n"
	if buf.String() != want {
		t.Errorf("indentWriter output:\n%q\nwant:\n%q", buf.String(), want)
	}
}

func TestLastLineAndNonEmptyLines(t *testing.T) {
	if got := lastLine("a\n\nb\nc\n"); got != "c" {
		t.Errorf("lastLine = %q", got)
	}
	lines := nonEmptyLines(" x \n\n y\n")
	if len(lines) != 2 || lines[0] != "x" || lines[1] != "y" {
		t.Errorf("nonEmptyLines = %v", lines)
	}
}

// TestBannerAndRuleAlignment guards the visual layout: with color disabled,
// every banner line and the summary rule must have the same display width.
func TestBannerAndRuleAlignment(t *testing.T) {
	var buf bytes.Buffer
	u := &ui{w: &buf, color: false}
	u.banner()
	u.rule("网关配置")
	out := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(out) != 5 {
		t.Fatalf("expected 5 lines, got %d: %q", len(out), out)
	}
	want := dispWidth(out[0])
	for i, line := range out {
		if got := dispWidth(line); got != want {
			t.Errorf("line %d width %d, want %d: %q", i, got, want, line)
		}
	}
	// Rule line must match the banner box width too.
	if got := dispWidth(out[4]); got != want {
		t.Errorf("rule width %d, want %d: %q", got, want, out[4])
	}
}
