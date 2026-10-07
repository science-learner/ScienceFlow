package tailer

import (
	"regexp"
	"strings"
)

// Multiline implements Filebeat-style multiline merging: it aggregates
// consecutive lines into a single event until a line matching the pattern
// starts a new event.
//
// negate = false: a line matching the pattern begins a NEW event.
// negate = true : a line matching the pattern is a CONTINUATION line.
//
// match  = after : continuation lines are appended after the header line.
// match  = before: continuation lines are appended and the matching line closes the event.
type Multiline struct {
	pattern *regexp.Regexp
	negate  bool
	match   string
	buf     []string
}

func NewMultiline(pattern string, negate bool, match string) (*Multiline, error) {
	if pattern == "" {
		return nil, nil
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, err
	}
	return NewMultilineFromRegex(re, negate, match), nil
}

// NewMultilineFromRegex builds a Multiline from an already-compiled regexp.
func NewMultilineFromRegex(re *regexp.Regexp, negate bool, match string) *Multiline {
	if re == nil {
		return nil
	}
	if match == "" {
		match = "after"
	}
	return &Multiline{pattern: re, negate: negate, match: match, buf: make([]string, 0, 8)}
}

// Process feeds a single line and returns the assembled message, if one is
// complete. It is not safe to call from multiple goroutines concurrently.
func (m *Multiline) Process(line string) (string, bool) {
	if m == nil {
		return line, true
	}
	isNewEvent := m.pattern.MatchString(line) != m.negate

	if m.match == "before" {
		if isNewEvent {
			m.buf = append(m.buf, line)
			return m.flush()
		}
		m.buf = append(m.buf, line)
		return "", false
	}

	// match == after
	if isNewEvent && len(m.buf) > 0 {
		out, _ := m.flush()
		m.buf = append(m.buf, line)
		return out, true
	}
	m.buf = append(m.buf, line)
	return "", false
}

// Flush emits a dangling partial event, if any (e.g. after a multiline timeout
// or when a harvester is shutting down).
func (m *Multiline) Flush() (string, bool) {
	if m == nil {
		return "", false
	}
	return m.flush()
}

func (m *Multiline) flush() (string, bool) {
	if len(m.buf) == 0 {
		return "", false
	}
	out := strings.Join(m.buf, "\n")
	m.buf = m.buf[:0]
	return out, true
}
