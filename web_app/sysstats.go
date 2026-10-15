package main

// SystemStats carries live resource usage of the local machine running the
// app: overall CPU load, memory usage and storage of the primary drive.
type SystemStats struct {
	CPUPercent       float64 `json:"cpu_percent"`
	MemoryUsedBytes  uint64  `json:"memory_used_bytes"`
	MemoryTotalBytes uint64  `json:"memory_total_bytes"`
	MemoryPercent    float64 `json:"memory_percent"`
	DiskUsedBytes    uint64  `json:"disk_used_bytes"`
	DiskTotalBytes   uint64  `json:"disk_total_bytes"`
	DiskPercent      float64 `json:"disk_percent"`
	DiskPath         string  `json:"disk_path"`
}

// GetSystemStats returns live resource usage of the local machine. Windows
// collects real counters (sysstats_windows.go); other platforms return zeroed
// values (sysstats_other.go) so server-mode deployments still build and run.
func (a *App) GetSystemStats() (SystemStats, error) {
	return platformSystemStats()
}
