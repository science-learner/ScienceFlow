package server

import (
	"archive/zip"
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ---- workspace transfer (upload / download) ----

// transferSession resolves the session workspace for a transfer request:
// token auth, session existence, ownership, workspace root. Returns the
// workspace root on success and has already written the HTTP error otherwise.
func (s *Server) transferSession(w http.ResponseWriter, r *http.Request) (string, bool) {
	user, ok := s.userFromRequest(w, r)
	if !ok {
		return "", false
	}
	sessionID := r.URL.Query().Get("session_id")
	if sessionID == "" {
		http.Error(w, "session_id query parameter is required", http.StatusBadRequest)
		return "", false
	}
	owner := s.sessions.User(sessionID)
	if owner == "" {
		http.Error(w, "session not found", http.StatusNotFound)
		return "", false
	}
	if s.auth != nil && owner != user {
		s.writeUnauthorized(w)
		return "", false
	}
	root, ok := s.sessionWorkspaceDir(sessionID)
	if !ok {
		http.Error(w, "workspace root not configured", http.StatusServiceUnavailable)
		return "", false
	}
	return root, true
}

// transferTargetDir resolves the destination directory for an upload. An
// empty rel means the workspace root itself; the directory is created if
// missing.
func (s *Server) transferTargetDir(root, rel string) (string, error) {
	if rel == "" {
		if err := os.MkdirAll(root, 0o755); err != nil {
			return "", err
		}
		return root, nil
	}
	dir, err := s.resolveWorkspacePath(root, rel)
	if err != nil {
		return "", err
	}
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		return "", fmt.Errorf("target %q is not a directory", rel)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// sanitizeUploadRel normalizes a client-provided relative path (multipart
// filename or zip entry). Returns "" when the path must be rejected:
// absolute paths, ".." components, and drive-letter prefixes are refused so
// the result always stays inside the target directory.
func sanitizeUploadRel(name string) string {
	rel := strings.ReplaceAll(name, "\\", "/")
	if strings.HasPrefix(rel, "/") {
		return ""
	}
	for _, comp := range strings.Split(rel, "/") {
		if comp == ".." {
			return ""
		}
	}
	rel = path.Clean(rel)
	if rel == "" || rel == "." || strings.Contains(rel, ":") {
		return ""
	}
	return rel
}

// handleWorkspaceFileUpload accepts multipart/form-data with one or more file
// parts and writes them into the session workspace. Each part's filename may
// carry a relative subpath (browser webkitdirectory style), e.g.
// "data/input.csv"; subdirectories are created as needed.
//
// Note: the standard mime/multipart parser strips directory components from
// FileName (filepath.Base), so the raw filename is re-read from the part's
// Content-Disposition header to preserve subpaths.
func (s *Server) handleWorkspaceFileUpload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	root, ok := s.transferSession(w, r)
	if !ok {
		return
	}
	target, err := s.transferTargetDir(root, r.URL.Query().Get("path"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.ws.UploadMaxBytes)
	mr, err := r.MultipartReader()
	if err != nil {
		http.Error(w, "invalid multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}

	type uploadedFile struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
	}
	uploaded := make([]uploadedFile, 0)
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			status, msg := transferReadErr(err)
			http.Error(w, msg, status)
			return
		}
		if part.FileName() == "" {
			continue // ignore non-file form fields
		}
		_, params, err := mime.ParseMediaType(part.Header.Get("Content-Disposition"))
		if err != nil {
			http.Error(w, "bad Content-Disposition: "+err.Error(), http.StatusBadRequest)
			return
		}
		rel := sanitizeUploadRel(params["filename"])
		if rel == "" {
			http.Error(w, fmt.Sprintf("invalid upload filename %q", params["filename"]), http.StatusBadRequest)
			return
		}
		dest, err := s.resolveWorkspacePath(target, rel)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		n, err := io.Copy(out, part)
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			status, msg := transferReadErr(err)
			http.Error(w, "write file: "+msg, status)
			return
		}
		relRoot, _ := filepath.Rel(root, dest)
		uploaded = append(uploaded, uploadedFile{Path: filepath.ToSlash(relRoot), Size: n})
	}
	if len(uploaded) == 0 {
		http.Error(w, "no file parts in multipart form", http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"root":     root,
		"uploaded": uploaded,
	})
}

// transferReadErr maps stream read failures to status codes: request bodies
// exceeding MaxBytesReader become 413, everything else 400.
func transferReadErr(err error) (int, string) {
	var mbe *http.MaxBytesError
	if errors.As(err, &mbe) {
		return http.StatusRequestEntityTooLarge, "request body too large"
	}
	return http.StatusBadRequest, err.Error()
}

// handleWorkspaceDirUpload accepts a zip archive (raw body, or the first file
// part of a multipart request) and extracts it into the workspace, with
// zip-slip protection and pre-validated extraction limits.
func (s *Server) handleWorkspaceDirUpload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	root, ok := s.transferSession(w, r)
	if !ok {
		return
	}
	target, err := s.transferTargetDir(root, r.URL.Query().Get("path"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.ws.ArchiveMaxBytes)
	var data []byte
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		mr, err := r.MultipartReader()
		if err != nil {
			http.Error(w, "invalid multipart form: "+err.Error(), http.StatusBadRequest)
			return
		}
		for {
			part, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				status, msg := transferReadErr(err)
				http.Error(w, msg, status)
				return
			}
			if part.FileName() == "" {
				continue
			}
			data, err = io.ReadAll(part)
			if err != nil {
				status, msg := transferReadErr(err)
				http.Error(w, msg, status)
				return
			}
			break
		}
		if data == nil {
			http.Error(w, "no file parts in multipart form", http.StatusBadRequest)
			return
		}
	} else {
		data, err = io.ReadAll(r.Body)
		if err != nil {
			status, msg := transferReadErr(err)
			http.Error(w, msg, status)
			return
		}
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		http.Error(w, "invalid zip archive: "+err.Error(), http.StatusBadRequest)
		return
	}

	var files, total uint64
	skipped := 0
	for _, f := range zr.File {
		name := sanitizeUploadRel(f.Name)
		if name == "" {
			http.Error(w, fmt.Sprintf("invalid zip entry path %q", f.Name), http.StatusBadRequest)
			return
		}
		if f.FileInfo().IsDir() {
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			skipped++
			continue
		}
		files++
		total += f.UncompressedSize64
	}
	if files > uint64(s.ws.ExtractMaxFiles) || total > uint64(s.ws.ExtractMaxBytes) {
		http.Error(w, fmt.Sprintf("archive too large: %d files / %d bytes exceeds limits", files, total), http.StatusRequestEntityTooLarge)
		return
	}

	extracted := 0
	var written uint64
	for _, f := range zr.File {
		name := sanitizeUploadRel(f.Name)
		if f.FileInfo().IsDir() {
			d, err := s.resolveWorkspacePath(target, name)
			if err == nil {
				_ = os.MkdirAll(d, 0o755)
			}
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			continue
		}
		dest, err := s.resolveWorkspacePath(target, name)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		rc, err := f.Open()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			rc.Close()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		n, err := io.Copy(out, rc)
		rc.Close()
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			http.Error(w, "extract file: "+err.Error(), http.StatusInternalServerError)
			return
		}
		extracted++
		written += uint64(n)
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"root":             root,
		"extracted":        extracted,
		"total_bytes":      written,
		"skipped_symlinks": skipped,
	})
}

// handleWorkspaceFileDownload streams a single workspace file as an
// attachment; Content-Type comes from the extension map with content sniffing
// as fallback. Range requests are supported via http.ServeContent.
func (s *Server) handleWorkspaceFileDownload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	root, ok := s.transferSession(w, r)
	if !ok {
		return
	}
	rel := r.URL.Query().Get("path")
	if rel == "" {
		http.Error(w, "path query parameter is required", http.StatusBadRequest)
		return
	}
	full, err := s.resolveWorkspacePath(root, rel)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	fi, err := os.Stat(full)
	if err != nil {
		http.Error(w, "not found: "+err.Error(), http.StatusNotFound)
		return
	}
	if fi.IsDir() {
		http.Error(w, "path is a directory; use GET /api/workspace/directory/download", http.StatusBadRequest)
		return
	}
	f, err := os.Open(full)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer f.Close()

	ct := extContentTypes[strings.ToLower(filepath.Ext(full))]
	if ct == "" {
		head := make([]byte, 512)
		n, _ := f.Read(head)
		ct = http.DetectContentType(head[:n])
		_, _ = f.Seek(0, io.SeekStart)
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{
		"filename": filepath.Base(full),
	}))
	http.ServeContent(w, r, filepath.Base(full), fi.ModTime(), f)
}

// zipEntry is one file or directory packed into a directory download.
type zipEntry struct {
	rel   string // workspace-root-relative, slash-separated
	abs   string
	dir   bool
	size  int64
	mtime time.Time
}

// handleWorkspaceDirDownload zips one directory (path) or several files and
// directories (paths, comma-separated) and streams the archive. Zip entries
// use workspace-root-relative paths so the archive can be re-uploaded via
// POST /api/workspace/directory/upload into the same root.
func (s *Server) handleWorkspaceDirDownload(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	root, ok := s.transferSession(w, r)
	if !ok {
		return
	}

	var items []string
	if ps := r.URL.Query().Get("paths"); ps != "" {
		for _, p := range strings.Split(ps, ",") {
			if p = strings.TrimSpace(p); p != "" {
				items = append(items, p)
			}
		}
	} else if p := r.URL.Query().Get("path"); p != "" {
		items = []string{p}
	}
	if len(items) == 0 {
		http.Error(w, "path or paths query parameter is required", http.StatusBadRequest)
		return
	}

	var entries []zipEntry
	var total int64
	truncated := false
	for _, rel := range items {
		full, err := s.resolveWorkspacePath(root, rel)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		fi, err := os.Lstat(full)
		if err != nil {
			http.Error(w, "not found: "+err.Error(), http.StatusNotFound)
			return
		}
		if fi.IsDir() {
			_ = filepath.Walk(full, func(p string, info os.FileInfo, err error) error {
				if err != nil {
					return nil
				}
				if info.IsDir() && info.Name() == ".git" {
					return filepath.SkipDir
				}
				relp, err := filepath.Rel(root, p)
				if err != nil || relp == "." {
					return nil
				}
				if info.Mode()&os.ModeSymlink != 0 {
					return nil
				}
				entries = append(entries, zipEntry{
					rel:   filepath.ToSlash(relp),
					abs:   p,
					dir:   info.IsDir(),
					size:  info.Size(),
					mtime: info.ModTime(),
				})
				if !info.IsDir() {
					total += info.Size()
				}
				if len(entries) > s.ws.ZipMaxEntries || total > s.ws.ZipMaxBytes {
					truncated = true
					return filepath.SkipAll
				}
				return nil
			})
		} else {
			if fi.Mode()&os.ModeSymlink != 0 {
				continue
			}
			entries = append(entries, zipEntry{
				rel:   filepath.ToSlash(rel),
				abs:   full,
				size:  fi.Size(),
				mtime: fi.ModTime(),
			})
			total += fi.Size()
		}
	}
	if len(entries) == 0 {
		http.Error(w, "nothing to download", http.StatusNotFound)
		return
	}

	archiveName := "workspace.zip"
	if len(items) == 1 {
		archiveName = path.Base(strings.TrimSuffix(filepath.ToSlash(items[0]), "/")) + ".zip"
	}

	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{
		"filename": archiveName,
	}))
	w.Header().Set("X-Archive-Truncated", strconv.FormatBool(truncated))

	zw := zip.NewWriter(w)
	for _, e := range entries {
		name := e.rel
		if e.dir {
			name += "/"
		}
		fw, err := zw.CreateHeader(&zip.FileHeader{
			Name:     name,
			Method:   zip.Deflate,
			Modified: e.mtime,
		})
		if err != nil {
			continue
		}
		if e.dir {
			continue
		}
		f, err := os.Open(e.abs)
		if err != nil {
			continue
		}
		_, _ = io.Copy(fw, f)
		_ = f.Close()
	}
	_ = zw.Close()
}
