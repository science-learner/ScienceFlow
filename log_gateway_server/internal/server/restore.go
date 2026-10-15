package server

import "log_gateway_server/internal/agent"

// RestoreSessions recreates in-memory sessions for historical per-user task
// directories under the workspace root. It is called once at startup so that
// after a gateway restart the session list, activation, SSE backfill and
// continued chat all keep working for previously created sessions (the agent
// workspace — including memory — is reused as-is).
//
// Restored sessions receive the owner's currently allowed sources; their
// displayed last_active comes from on-disk timestamps.
func (s *Server) RestoreSessions() int {
	if s.agent == nil {
		return 0
	}
	root, err := s.agent.RootPath()
	if err != nil {
		if s.logger != nil {
			s.logger.Printf("restore sessions: %v", err)
		}
		return 0
	}
	historical, err := agent.DiscoverSessions(root)
	if err != nil {
		if s.logger != nil {
			s.logger.Printf("restore sessions: %v", err)
		}
		return 0
	}

	restored := 0
	for _, h := range historical {
		if s.sessions.Restore(h.ID, h.User, h.Name, s.allowedSources(h.User), h.LastActive) {
			restored++
		}
	}
	return restored
}
