package agent

import "sync"

// ringBuffer is a fixed-capacity byte buffer that keeps only the most recent
// `cap` bytes written, used to retain a tail of agent stdout/stderr without
// unbounded memory growth.
type ringBuffer struct {
	mu  sync.Mutex
	buf []byte
	cap int
}

func newRingBuffer(capBytes int) *ringBuffer {
	return &ringBuffer{cap: capBytes}
}

func (r *ringBuffer) write(p []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf = append(r.buf, p...)
	if len(r.buf) > r.cap {
		r.buf = r.buf[len(r.buf)-r.cap:]
	}
}

// tail returns up to maxBytes of the most recent content.
func (r *ringBuffer) tail(maxBytes int) []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	if maxBytes <= 0 || maxBytes >= len(r.buf) {
		out := make([]byte, len(r.buf))
		copy(out, r.buf)
		return out
	}
	start := len(r.buf) - maxBytes
	out := make([]byte, maxBytes)
	copy(out, r.buf[start:])
	return out
}
