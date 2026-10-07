package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

// SystemStats carries live resource usage of the local machine running the
// desktop app: overall CPU load, memory usage, and storage of the Windows
// system drive (normally C:\).
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

var (
	cpuSampleMu    sync.Mutex
	cpuLastIdle    uint64
	cpuLastTotal   uint64
	cpuHasLastTime bool
)

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procGetSystemTimes       = kernel32.NewProc("GetSystemTimes")
	procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")
)

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

func filetimeValue(ft *syscall.Filetime) uint64 {
	return uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime)
}

// readCPUTimes samples the system-wide CPU idle/total counters (100ns units).
// kernelTime already includes idle time.
func readCPUTimes() (idle uint64, total uint64) {
	var idleTime, kernelTime, userTime syscall.Filetime
	ret, _, _ := procGetSystemTimes.Call(
		uintptr(unsafe.Pointer(&idleTime)),
		uintptr(unsafe.Pointer(&kernelTime)),
		uintptr(unsafe.Pointer(&userTime)),
	)
	if ret == 0 {
		return 0, 0
	}
	idle = filetimeValue(&idleTime)
	total = filetimeValue(&kernelTime) + filetimeValue(&userTime)
	return idle, total
}

// cpuPercentSince returns the CPU busy percentage since the previous sample
// and stores the current one. The first call primes the baseline with a short
// pause so the value is meaningful right away.
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
	var status memoryStatusEx
	status.Length = uint32(unsafe.Sizeof(status))
	ret, _, _ := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&status)))
	if ret == 0 {
		return 0, 0, 0
	}
	used = status.TotalPhys - status.AvailPhys
	total = status.TotalPhys
	if total > 0 {
		percent = float64(used) / float64(total) * 100
	}
	return used, total, percent
}

// systemDrivePath returns the root path of the Windows system drive (the
// drive holding the OS, normally C:\).
func systemDrivePath() string {
	if drive := os.Getenv("SystemDrive"); drive != "" {
		return drive + string(os.PathSeparator)
	}
	return `C:\`
}

func readDisk(path string) (used, total uint64, percent float64) {
	var freeAvailable, totalBytes, totalFree uint64
	utf16Path, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, 0, 0
	}
	if err := windows.GetDiskFreeSpaceEx(utf16Path, &freeAvailable, &totalBytes, &totalFree); err != nil {
		return 0, 0, 0
	}
	used = totalBytes - freeAvailable
	total = totalBytes
	if total > 0 {
		percent = float64(used) / float64(total) * 100
	}
	return used, total, percent
}

// GetSystemStats returns live resource usage of the local machine. Non-Windows
// platforms return zeroed values (the desktop build targets Windows).
func (a *App) GetSystemStats() (SystemStats, error) {
	stats := SystemStats{}
	if runtime.GOOS != "windows" {
		return stats, nil
	}
	stats.CPUPercent = cpuPercentSince()
	stats.MemoryUsedBytes, stats.MemoryTotalBytes, stats.MemoryPercent = readMemory()
	stats.DiskPath = systemDrivePath()
	stats.DiskUsedBytes, stats.DiskTotalBytes, stats.DiskPercent = readDisk(stats.DiskPath)
	if stats.MemoryTotalBytes == 0 && stats.DiskTotalBytes == 0 {
		return stats, fmt.Errorf("system stats unavailable")
	}
	return stats, nil
}
