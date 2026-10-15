package main

import (
	"bytes"
	"io"
	"log"
	"os"
)

// quietTransportLogs filters the known-benign Go net/http transport message
// emitted when a reused keep-alive connection receives an unsolicited response
// — typically a bare "HTTP/1.1 400 Bad Request" written by an intermediary
// (WSL port relay, Docker proxy, Cloudflare tunnel) when the origin closes an
// idle connection. The transport already discards that connection and retries
// safely, so the log line is pure noise.
type quietTransportLogs struct{ out io.Writer }

func (w quietTransportLogs) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("Unsolicited response received on idle HTTP channel")) {
		return len(p), nil
	}
	return w.out.Write(p)
}

func init() {
	log.SetOutput(quietTransportLogs{out: os.Stderr})
}
