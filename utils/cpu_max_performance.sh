#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root"
  exit 1
fi

echo "========================================"
echo "    Configuring System Performance      "
echo "========================================"

# 1. Set CPU Governor and EPP
echo "[CPU] Setting governors to 'performance'..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq; do
    if [ -f "$cpu/scaling_governor" ]; then
        echo performance > "$cpu/scaling_governor"
    fi
    
    # Energy Performance Preference (EPP) for modern Intel/AMD CPUs
    if [ -f "$cpu/energy_performance_preference" ]; then
        echo performance > "$cpu/energy_performance_preference"
    fi
done

# 2. Force Max Frequency
echo "[CPU] Forcing max scaling frequency..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq; do
    if [ -f "$cpu/cpuinfo_max_freq" ] && [ -f "$cpu/scaling_max_freq" ]; then
        cat "$cpu/cpuinfo_max_freq" > "$cpu/scaling_max_freq"
    fi
done

# 3. Report CPU Status (CPU 0 sample)
echo "----------------------------------------"
echo "CPU 0 Status:"
echo "  Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "  EPP:      $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"
echo "  Freq:     $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null) / $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null) KHz"

# 4. Set GPU Performance (ROCm)
echo "----------------------------------------"
if command -v rocm-smi &> /dev/null; then
    echo "[GPU] Setting ROCm performance level to 'high'..."
    rocm-smi --setperflevel high
    echo "[GPU] Current Performance Level:"
    rocm-smi --showperflevel
else
    echo "[GPU] rocm-smi not found, skipping GPU configuration."
fi

echo "========================================"
echo "Done. System configured for MAX performance."
