//go:build linux

package main

import (
	"fmt"
	"os"
	"sync"
	"syscall"
	"time"
)

var (
	cpuSampleMu    sync.Mutex
	cpuLastIdle    uint64
	cpuLastTotal   uint64
	cpuHasLastTime bool
)

// readCPUTimes samples the system-wide CPU idle/total jiffies from /proc/stat.
func readCPUTimes() (idle uint64, total uint64) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, 0
	}
	idle, total, ok := parseProcStatCPU(string(data))
	if !ok {
		return 0, 0
	}
	return idle, total
}

// cpuPercentSince returns the CPU busy percentage since the previous sample
// and stores the current one. The first call primes the baseline with a short
// pause so the value is meaningful right away. Mirrors sysstats_windows.go.
func cpuPercentSince() float64 {
	cpuSampleMu.Lock()
	defer cpuSampleMu.Unlock()
	idle, total := readCPUTimes()
	if !cpuHasLastTime {
		cpuHasLastTime = true
		cpuLastIdle, cpuLastTotal = idle, total
		time.Sleep(300 * time.Millisecond)
		idle, total = readCPUTimes()
	}
	idleDelta := idle - cpuLastIdle
	totalDelta := total - cpuLastTotal
	cpuLastIdle, cpuLastTotal = idle, total
	if totalDelta == 0 {
		return 0
	}
	percent := float64(totalDelta-idleDelta) / float64(totalDelta) * 100
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}
	return percent
}

func readMemory() (used, total uint64, percent float64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0, 0
	}
	total, available, ok := parseMeminfo(string(data))
	if !ok || total == 0 {
		return 0, 0, 0
	}
	used = total - available
	percent = float64(used) / float64(total) * 100
	return used, total, percent
}

// systemRootPath returns the mount point whose usage is reported, mirroring
// the Windows build's system drive (C:\).
func systemRootPath() string {
	return "/"
}

func readDisk(path string) (used, total uint64, percent float64) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, 0, 0
	}
	total = uint64(st.Blocks) * uint64(st.Bsize)
	freeAvailable := uint64(st.Bavail) * uint64(st.Bsize)
	if freeAvailable > total {
		freeAvailable = total
	}
	used = total - freeAvailable
	if total > 0 {
		percent = float64(used) / float64(total) * 100
	}
	return used, total, percent
}

func platformSystemStats() (SystemStats, error) {
	stats := SystemStats{}
	stats.CPUPercent = cpuPercentSince()
	stats.MemoryUsedBytes, stats.MemoryTotalBytes, stats.MemoryPercent = readMemory()
	stats.DiskPath = systemRootPath()
	stats.DiskUsedBytes, stats.DiskTotalBytes, stats.DiskPercent = readDisk(stats.DiskPath)
	if stats.MemoryTotalBytes == 0 && stats.DiskTotalBytes == 0 {
		return stats, fmt.Errorf("system stats unavailable")
	}
	return stats, nil
}
