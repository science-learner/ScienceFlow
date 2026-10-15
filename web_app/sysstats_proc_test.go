package main

import "testing"

// sampleStatLine is the real aggregate line captured from a Linux container
// (x86_64, kernel 6.x) via `head -1 /proc/stat`.
const sampleStatLine = "cpu  708084 0 408356 236301976 4594 0 46568 0 0 0"

func TestParseProcStatCPU(t *testing.T) {
	sample := sampleStatLine + "\ncpu0 50 10 15 200 5 2 3 0 0 0\nintr 12345\nctxt 67890\n"
	idle, total, ok := parseProcStatCPU(sample)
	if !ok {
		t.Fatal("expected ok")
	}
	if idle != 236301976+4594 {
		t.Fatalf("idle = %d, want %d", idle, 236301976+4594)
	}
	if total != 708084+0+408356+236301976+4594+0+46568+0 {
		t.Fatalf("total = %d, want %d", total, 708084+0+408356+236301976+4594+0+46568+0)
	}
}

func TestParseProcStatCPUExcludesGuest(t *testing.T) {
	idle, total, ok := parseProcStatCPU("cpu 10 0 0 90 0 0 0 0 100 100\n")
	if !ok {
		t.Fatal("expected ok")
	}
	if idle != 90 {
		t.Fatalf("idle = %d, want 90", idle)
	}
	if total != 100 {
		t.Fatalf("total = %d, want 100 (guest times must not be double counted)", total)
	}
}

func TestParseProcStatCPURejectsBadInput(t *testing.T) {
	cases := map[string]string{
		"no aggregate line": "cpu0 1 2 3 4 5 6 7 8\nintr 12345\n",
		"too few fields":    "cpu 1 2 3\n",
		"non numeric":       "cpu 1 2 x 4 5\n",
		"empty":             "",
	}
	for name, data := range cases {
		if _, _, ok := parseProcStatCPU(data); ok {
			t.Fatalf("%s: expected ok=false", name)
		}
	}
}

func TestParseMeminfo(t *testing.T) {
	sample := "MemTotal:        7991476 kB\nMemFree:         5567612 kB\nMemAvailable:    6109452 kB\nBuffers:            1140 kB\nCached:           437128 kB\n"
	total, available, ok := parseMeminfo(sample)
	if !ok {
		t.Fatal("expected ok")
	}
	if total != 7991476*1024 {
		t.Fatalf("total = %d, want %d", total, 7991476*1024)
	}
	if available != 6109452*1024 {
		t.Fatalf("available = %d, want %d", available, 6109452*1024)
	}
}

func TestParseMeminfoFallbackWithoutMemAvailable(t *testing.T) {
	sample := "MemTotal:        1000000 kB\nMemFree:          200000 kB\nBuffers:           50000 kB\nCached:           300000 kB\nSReclaimable:      30000 kB\nShmem:             10000 kB\n"
	total, available, ok := parseMeminfo(sample)
	if !ok {
		t.Fatal("expected ok")
	}
	if total != 1000000*1024 {
		t.Fatalf("total = %d, want %d", total, 1000000*1024)
	}
	want := uint64(200000+50000+300000+30000-10000) * 1024
	if available != want {
		t.Fatalf("available = %d, want %d", available, want)
	}
}

func TestParseMeminfoRejectsMissingTotal(t *testing.T) {
	if _, _, ok := parseMeminfo("MemFree: 100 kB\n"); ok {
		t.Fatal("expected ok=false without MemTotal")
	}
}

func TestParseMeminfoClampsAvailable(t *testing.T) {
	total, available, ok := parseMeminfo("MemTotal: 1000 kB\nMemAvailable: 2000 kB\n")
	if !ok {
		t.Fatal("expected ok")
	}
	if available != total {
		t.Fatalf("available = %d, want clamp to total %d", available, total)
	}
}
