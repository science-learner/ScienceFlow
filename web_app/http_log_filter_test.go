package main

import (
	"bufio"
	"bytes"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestQuietTransportLogsFiltersOnlyUnsolicitedResponse(t *testing.T) {
	var buf bytes.Buffer
	w := quietTransportLogs{out: &buf}

	noise := []byte("2026/10/13 13:57:19 Unsolicited response received on idle HTTP channel starting with \"HTTP/1.1 400 Bad Request\\r\\n\\r\\n\"; err=<nil>\n")
	if n, err := w.Write(noise); err != nil || n != len(noise) {
		t.Fatalf("noise line: n=%d err=%v", n, err)
	}
	if buf.Len() != 0 {
		t.Fatalf("noise line was not filtered: %q", buf.String())
	}

	keep := []byte("2026/10/13 13:57:19 ScienceFlow (server): gateway=http://127.0.0.1:52001\n")
	if _, err := w.Write(keep); err != nil {
		t.Fatalf("keep line: %v", err)
	}
	if buf.String() != string(keep) {
		t.Fatalf("keep line altered: %q", buf.String())
	}
}

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// poisonServer answers every request with a valid 200 and then, once the
// connection goes idle, writes a bare "HTTP/1.1 400 Bad Request" onto it —
// exactly what intermediaries (WSL relay, Docker proxy, Cloudflare tunnel)
// do to stale keep-alive connections.
func poisonServer(t *testing.T) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(c net.Conn) {
				defer c.Close()
				br := bufio.NewReader(c)
				for {
					if _, err := http.ReadRequest(br); err != nil {
						return
					}
					c.Write([]byte("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"))
					time.Sleep(150 * time.Millisecond)
					c.Write([]byte("HTTP/1.1 400 Bad Request\r\n\r\n"))
					time.Sleep(600 * time.Millisecond)
				}
			}(conn)
		}
	}()
	return "http://" + ln.Addr().String() + "/", func() { ln.Close() }
}

func exerciseIdleReuse(t *testing.T, url string) {
	t.Helper()
	transport := &http.Transport{}
	client := &http.Client{Transport: transport}
	defer transport.CloseIdleConnections()
	for i := 0; i < 2; i++ {
		resp, err := client.Get(url)
		if err != nil {
			t.Fatalf("request %d: %v", i+1, err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		time.Sleep(250 * time.Millisecond)
	}
	time.Sleep(400 * time.Millisecond)
}

func TestStdlibUnsolicitedResponseIsFiltered(t *testing.T) {
	url, closeServer := poisonServer(t)
	defer closeServer()
	defer log.SetOutput(quietTransportLogs{out: os.Stderr})

	// Control: without the filter the stdlib emits the message.
	control := &syncBuffer{}
	log.SetOutput(control)
	exerciseIdleReuse(t, url)
	if !strings.Contains(control.String(), "Unsolicited response received on idle HTTP channel") {
		t.Fatalf("scenario did not trigger the stdlib log: %q", control.String())
	}

	// With the filter the message is silenced while other lines pass through.
	filtered := &syncBuffer{}
	log.SetOutput(quietTransportLogs{out: filtered})
	log.Printf("keep-me")
	exerciseIdleReuse(t, url)
	if strings.Contains(filtered.String(), "Unsolicited response") {
		t.Fatalf("filtered logger let the message through: %q", filtered.String())
	}
	if !strings.Contains(filtered.String(), "keep-me") {
		t.Fatalf("filtered logger dropped unrelated output: %q", filtered.String())
	}
}
