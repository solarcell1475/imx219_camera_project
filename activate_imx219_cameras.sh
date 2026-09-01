#!/usr/bin/env bash

# Select an existing Jetson-IO boot entry containing the camera overlay.
# This script does not create an overlay; configure one with Jetson-IO first.

set -Eeuo pipefail

BOOT_CONFIG="/boot/extlinux/extlinux.conf"
CHECK_ONLY=false
REBOOT_NOW=false

usage() {
    cat <<'EOF'
Usage: sudo ./activate_imx219_cameras.sh [--check] [--reboot]

  --check   Validate the Jetson-IO boot entry without changing anything.
  --reboot  Reboot immediately after a successful configuration change.
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
        --check) CHECK_ONLY=true ;;
        --reboot) REBOOT_NOW=true ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

echo "IMX219-83 Stereo Camera Boot Configuration"
echo "==========================================="

if ! is_jetson; then
    echo "ERROR: This does not appear to be an NVIDIA Jetson." >&2
    echo "Run this script only on the target Jetson Orin Nano." >&2
    exit 1
fi

if [[ ! -r "$BOOT_CONFIG" ]]; then
    echo "ERROR: Cannot read $BOOT_CONFIG." >&2
    exit 1
fi

jetson_io_entry="$(
    awk '
        $1 == "LABEL" {
            if (found) exit
            if ($2 == "JetsonIO") found = 1
        }
        found { print }
    ' "$BOOT_CONFIG"
)"

if [[ -z "$jetson_io_entry" ]]; then
    echo "ERROR: No JetsonIO boot entry exists in $BOOT_CONFIG." >&2
    echo "Create the dual IMX219 overlay with Jetson-IO, then rerun --check." >&2
    exit 1
fi

if ! grep -qi 'imx219' <<< "$jetson_io_entry"; then
    echo "ERROR: The JetsonIO boot entry does not reference an IMX219 overlay." >&2
    echo "Reconfigure the dual IMX219 overlay with Jetson-IO before selecting it." >&2
    exit 1
fi

echo "Found JetsonIO boot entry:"
grep -E 'LABEL|MENU LABEL|OVERLAYS|FDT' <<< "$jetson_io_entry" || true

current_default="$(awk '$1 == "DEFAULT" { print $2; exit }' "$BOOT_CONFIG")"
if [[ -z "$current_default" ]]; then
    echo "ERROR: No DEFAULT boot entry exists in $BOOT_CONFIG." >&2
    exit 1
fi

echo "Current default boot entry: $current_default"

if "$CHECK_ONLY"; then
    if [[ "$current_default" == "JetsonIO" ]]; then
        echo "PASS: JetsonIO is already the default boot entry."
    else
        echo "READY: JetsonIO exists but is not the default boot entry."
        echo "Run: sudo ./activate_imx219_cameras.sh"
    fi
    exit 0
fi

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: Configuration changes require root privileges." >&2
    echo "Run: sudo ./activate_imx219_cameras.sh" >&2
    exit 1
fi

if [[ "$current_default" == "JetsonIO" ]]; then
    echo "No change needed: JetsonIO is already the default boot entry."
else
    backup="${BOOT_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "Creating backup: $backup"
    cp --preserve=all "$BOOT_CONFIG" "$backup"

    sed -i -E 's/^DEFAULT[[:space:]]+.*/DEFAULT JetsonIO/' "$BOOT_CONFIG"

    if [[ "$(awk '$1 == "DEFAULT" { print $2; exit }' "$BOOT_CONFIG")" != "JetsonIO" ]]; then
        echo "ERROR: Failed to select JetsonIO; restoring backup." >&2
        cp --preserve=all "$backup" "$BOOT_CONFIG"
        exit 1
    fi
    echo "PASS: JetsonIO is now the default boot entry."
fi

echo
echo "A reboot is required before testing the cameras."
echo "After reboot, run: ./test_imx219_cameras.sh --dual-smoke"

if "$REBOOT_NOW"; then
    echo "Rebooting now..."
    reboot
else
    echo "Reboot when ready: sudo reboot"
fi
