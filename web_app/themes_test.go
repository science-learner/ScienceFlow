package main

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestThemeBackgrounds(t *testing.T) {
	dir := t.TempDir()

	// 1x1 transparent PNG.
	png, err := base64.StdEncoding.DecodeString(
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "beta.png"), png, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "alpha.JPG"), png, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub.png"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := listThemeBackgroundsIn(dir)
	if len(got) != 2 || got[0] != "alpha.JPG" || got[1] != "beta.png" {
		t.Fatalf("list = %v, want [alpha.JPG beta.png]", got)
	}

	data := readThemeBackgroundIn(dir, "beta.png")
	if !strings.HasPrefix(data, "data:image/png;base64,") {
		t.Fatalf("read beta.png = %q, want png data URI", data)
	}
	if readThemeBackgroundIn(dir, "alpha.JPG") == "" {
		t.Fatal("uppercase extension should be accepted")
	}
	if readThemeBackgroundIn(dir, "notes.txt") != "" {
		t.Fatal("unsupported extension should be rejected")
	}
	if readThemeBackgroundIn(dir, "../beta.png") != "" {
		t.Fatal("path traversal should be rejected")
	}
	if readThemeBackgroundIn(dir, "missing.png") != "" {
		t.Fatal("missing file should be rejected")
	}
	if readThemeBackgroundIn(dir, "sub.png") != "" {
		t.Fatal("directories should be rejected")
	}
}
