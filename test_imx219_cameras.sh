#!/usr/bin/env bash

# Headless diagnostics and finite capture tests for dual IMX219 CSI cameras.

set -uo pipefail

MODE="diagnose"
BUFFERS=60
TIMEOUT_SECONDS=20
CAPS="video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1"

usage() {
    cat <<'EOF'
Usage: ./test_imx219_cameras.sh [MODE]

Modes:
  --diagnose     Report platform, device, plugin, and kernel status (default).
  --smoke        Capture a finite headless stream from each sensor in sequence.
  --dual-smoke   Capture finite headless streams from both sensors concurrently.
  --all          Run diagnostics, sequential smoke tests, and the dual test.

The smoke tests require Jetson hardware, NVArgus, and the camera overlay.
EOF
}

is_jetson() {
    if [[ -r /etc/nv_tegra_release ]]; then
        return 0
    fi

    [[ -r /proc/device-tree/compatible ]] &&
        tr '\0' '\n' < /proc/device-tree/compatible | grep -qiE 'nvidia|tegra'
}

for argument in "$@"; do
    case "$argument" in
        --diagnose) MODE="diagnose" ;;
        --smoke) MODE="smoke" ;;
        --dual-smoke) MODE="dual-smoke" ;;
        --all) MODE="all" ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

diagnose() {
    local failed=0

    echo "IMX219-83 Stereo Camera Diagnostics"
    echo "==================================="

    if is_jetson; then
        echo "PASS: NVIDIA Jetson platform detected."
    else
        echo "FAIL: This host is not an NVIDIA Jetson."
        failed=1
    fi

    shopt -s nullglob
    local video_devices=(/dev/video*)
    shopt -u nullglob
    if ((${#video_devices[@]})); then
        echo "Video device nodes:"
        printf '  %s\n' "${video_devices[@]}"
    else
        echo "WARN: No /dev/video* device nodes found."
        failed=1
    fi

    if command -v v4l2-ctl >/dev/null 2>&1; then
        echo
        echo "V4L2 device inventory:"
        v4l2-ctl --list-devices || failed=1
    else
        echo "WARN: v4l2-ctl is missing (install package v4l-utils)."
    fi

    if command -v gst-launch-1.0 >/dev/null 2>&1 &&
        command -v gst-inspect-1.0 >/dev/null 2>&1; then
        if gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1; then
            echo "PASS: nvarguscamerasrc GStreamer plugin is available."
        else
            echo "FAIL: nvarguscamerasrc GStreamer plugin is unavailable."
            failed=1
        fi
    else
        echo "FAIL: GStreamer command-line tools are unavailable."
        failed=1
    fi

    if command -v systemctl >/dev/null 2>&1 &&
        systemctl is-active --quiet nvargus-daemon.service; then
        echo "PASS: nvargus-daemon.service is active."
    else
        echo "WARN: nvargus-daemon.service is not reported active."
    fi

    echo
    echo "Recent IMX219 kernel messages:"
    if ! dmesg 2>/dev/null | grep -i imx219 | tail -n 20; then
        echo "  No readable IMX219 kernel messages found."
    fi

    if ((failed)); then
        echo
        echo "Diagnostics found blocking issues."
        return 1
    fi

    echo
    echo "Diagnostics passed. Prove capture with --dual-smoke."
}

require_capture_runtime() {
    if ! is_jetson; then
        echo "ERROR: Capture tests must run on the Jetson." >&2
        return 1
    fi
    if ! command -v timeout >/dev/null 2>&1 ||
        ! command -v gst-launch-1.0 >/dev/null 2>&1 ||
        ! command -v gst-inspect-1.0 >/dev/null 2>&1 ||
        ! gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1; then
        echo "ERROR: timeout, GStreamer, or nvarguscamerasrc is unavailable." >&2
        return 1
    fi
}

smoke_sensor() {
    local sensor_id="$1"

    echo "Testing sensor-id=$sensor_id ($BUFFERS buffers, headless)..."
    if timeout "${TIMEOUT_SECONDS}s" gst-launch-1.0 -q \
        nvarguscamerasrc sensor-id="$sensor_id" num-buffers="$BUFFERS" ! \
        "$CAPS" ! queue ! fakesink sync=false; then
        echo "PASS: sensor-id=$sensor_id produced a finite stream."
    else
        local status=$?
        echo "FAIL: sensor-id=$sensor_id capture exited with status $status." >&2
        return "$status"
    fi
}

smoke_sequential() {
    require_capture_runtime || return 1
    smoke_sensor 0 || return 1
    smoke_sensor 1
}

smoke_dual() {
    require_capture_runtime || return 1

    echo "Testing sensor-id=0 and sensor-id=1 concurrently..."
    if timeout "${TIMEOUT_SECONDS}s" gst-launch-1.0 -q \
        nvarguscamerasrc sensor-id=0 num-buffers="$BUFFERS" ! \
        "$CAPS" ! queue ! fakesink sync=false \
        nvarguscamerasrc sensor-id=1 num-buffers="$BUFFERS" ! \
        "$CAPS" ! queue ! fakesink sync=false; then
        echo "PASS: Both sensors produced concurrent finite streams."
    else
        local status=$?
        echo "FAIL: Concurrent capture exited with status $status." >&2
        return "$status"
    fi
}

case "$MODE" in
    diagnose) diagnose ;;
    smoke) smoke_sequential ;;
    dual-smoke) smoke_dual ;;
    all)
        result=0
        diagnose || result=1
        smoke_sequential || result=1
        smoke_dual || result=1
        exit "$result"
        ;;
esac
