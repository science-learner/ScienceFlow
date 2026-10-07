package main

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// themeBackgroundsDir is the drop-in folder, relative to the program
// directory, that holds user-provided Agent Map background images. The
// settings dropdown lists its contents by file name.
const themeBackgroundsDir = "themes/backgrounds"

// maxThemeBackgroundBytes guards the webview against huge data URIs.
const maxThemeBackgroundBytes = 32 << 20 // 32 MiB

var themeBackgroundMIME = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".webp": "image/webp",
	".gif":  "image/gif",
	".bmp":  "image/bmp",
}

// resolveThemeBackgroundsDir returns <program dir>/themes/backgrounds,
// falling back to the working directory (for `wails3 dev`). The directory is
// created on first use so users have a clear place to drop images. Empty
// string means no usable location.
func resolveThemeBackgroundsDir() string {
	if exe, err := os.Executable(); err == nil {
		if dir := ensureDir(filepath.Join(filepath.Dir(exe), "themes", "backgrounds")); dir != "" {
			return dir
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		if dir := ensureDir(filepath.Join(cwd, "themes", "backgrounds")); dir != "" {
			return dir
		}
	}
	return ""
}

func ensureDir(dir string) string {
	if st, err := os.Stat(dir); err == nil && st.IsDir() {
		return dir
	}
	if err := os.MkdirAll(dir, 0o755); err == nil {
		return dir
	}
	return ""
}

// listThemeBackgroundsIn returns supported image file names in dir, sorted.
func listThemeBackgroundsIn(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []string{}
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if _, ok := themeBackgroundMIME[strings.ToLower(filepath.Ext(name))]; !ok {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// readThemeBackgroundIn loads dir/name as a data URI. Only bare file names
// with a supported image extension are accepted.
func readThemeBackgroundIn(dir, name string) string {
	if dir == "" || name == "" {
		return ""
	}
	if name != filepath.Base(name) || name == "." || name == ".." {
		return ""
	}
	mime, ok := themeBackgroundMIME[strings.ToLower(filepath.Ext(name))]
	if !ok {
		return ""
	}
	full := filepath.Join(dir, name)
	fi, err := os.Stat(full)
	if err != nil || fi.IsDir() || fi.Size() == 0 || fi.Size() > maxThemeBackgroundBytes {
		return ""
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return ""
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(b)
}

// ListThemeBackgrounds returns the Agent Map background images available in
// <program dir>/themes/backgrounds. The option label equals the file name.
func (a *App) ListThemeBackgrounds() []string {
	return listThemeBackgroundsIn(resolveThemeBackgroundsDir())
}

// ReadThemeBackground returns one background image as a data URI so the
// webview can render files that live outside the bundled frontend assets.
// An empty string signals an unknown, unreadable or oversized image.
func (a *App) ReadThemeBackground(name string) string {
	return readThemeBackgroundIn(resolveThemeBackgroundsDir(), name)
}
