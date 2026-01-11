#!/bin/bash

# MAXN Power Mode Setup Script
# =============================
# Sets Jetson Orin Nano to maximum performance mode

set -e

echo "========================================="
echo "Jetson Orin Nano MAXN Power Mode Setup"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

# Set MAXN power mode
echo "Setting power mode to MAXN (Maximum Performance)..."
nvpmodel -m 0

# Verify power mode
echo ""
echo "Current power mode:"
nvpmodel -q

# Set GPU to maximum clock
echo ""
echo "Setting GPU to maximum clock..."
jetson_clocks

# Verify GPU clocks
echo ""
echo "GPU clock status:"
cat /sys/kernel/debug/clock/gpu/rate 2>/dev/null || echo "GPU clock info not available"

echo ""
echo "========================================="
echo "MAXN power mode configured!"
echo "========================================="
echo ""
echo "Note: These settings may reset after reboot."
echo "To make persistent, add to startup scripts."
