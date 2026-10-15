package main

import (
	"strconv"
	"strings"
)

// parseProcStatCPU extracts aggregate idle and total jiffies from the first
// "cpu " line of /proc/stat. Guest times are capped off: the kernel already
// counts guest/guest_nice inside user/nice, so including them would double
// count.
func parseProcStatCPU(data string) (idle uint64, total uint64, ok bool) {
	for _, line := range strings.Split(data, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 5 || fields[0] != "cpu" {
			continue
		}
		values := make([]uint64, 0, len(fields)-1)
		for _, f := range fields[1:] {
			v, err := strconv.ParseUint(f, 10, 64)
			if err != nil {
				return 0, 0, false
			}
			values = append(values, v)
		}
		// user nice system idle iowait irq softirq steal [guest guest_nice]
		n := len(values)
		if n > 8 {
			n = 8
		}
		for _, v := range values[:n] {
			total += v
		}
		idle = values[3]
		if n > 4 {
			idle += values[4]
		}
		return idle, total, true
	}
	return 0, 0, false
}

// parseMeminfo extracts MemTotal and MemAvailable (in bytes) from
// /proc/meminfo. Kernels without MemAvailable (pre-3.14) fall back to the
// documented approximation free + buffers + cached + SReclaimable - Shmem.
func parseMeminfo(data string) (total uint64, available uint64, ok bool) {
	var free, buffers, cached, sreclaimable, shmem uint64
	for _, line := range strings.Split(data, "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		fields := strings.Fields(value)
		if len(fields) == 0 {
			continue
		}
		kb, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		bytes := kb * 1024
		switch key {
		case "MemTotal":
			total = bytes
		case "MemAvailable":
			available = bytes
		case "MemFree":
			free = bytes
		case "Buffers":
			buffers = bytes
		case "Cached":
			cached = bytes
		case "SReclaimable":
			sreclaimable = bytes
		case "Shmem":
			shmem = bytes
		}
	}
	if total == 0 {
		return 0, 0, false
	}
	if available == 0 {
		available = free + buffers + cached + sreclaimable
		if available > shmem {
			available -= shmem
		}
	}
	if available > total {
		available = total
	}
	return total, available, true
}
