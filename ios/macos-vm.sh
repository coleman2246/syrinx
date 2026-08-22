#!/usr/bin/env bash
# The macOS build VM, for building the iOS app from Linux.
#
# Xcode runs nowhere else, so a guest is the only way to build the phone app
# without a second machine. This is an OSX-KVM style invocation with the
# defaults corrected for an AMD host and a Sequoia guest; see docs/ios.md.
#
# It expects a directory holding the images OSX-KVM produces:
#
#   OVMF_CODE_4M.fd, OVMF_VARS-1920x1080.fd    firmware
#   OpenCore/OpenCore.qcow2                    bootloader
#   mac_hdd_ng.img                             the installed system
#   BaseSystem.img                             installer, only while installing
#
# None of those are in this repository: they are large, and they are Apple's.
# Note also that Apple's licence permits macOS only on Apple hardware, so this
# is for building on a Mac you own, or a decision you are making knowingly.
#
#   VM_DIR=/mnt/winssd/macos-vm ios/macos-vm.sh
#
set -euo pipefail

VM_DIR="${VM_DIR:-$(pwd)}"
RAM_MIB="${RAM_MIB:-12288}"     # Xcode is happier with 12 GiB than 8
CORES="${CORES:-4}"
THREADS="${THREADS:-8}"         # half of a 16-thread host
SSH_PORT="${SSH_PORT:-2222}"
VNC_DISPLAY="${VNC_DISPLAY:-1}" # viewer connects to localhost:5901
MONITOR="${MONITOR:-/tmp/syrinx-macvm.sock}"
MAC="${MAC:-52:54:00:c9:18:27}"

# isa-applesmc needs Apple's OSK, which is a string from Apple's firmware and
# not something to commit to a public repository. It is widely published and
# OSX-KVM will tell you it; put it in the environment or in an ignored file
# next to the images.
if [ -z "${APPLESMC_OSK:-}" ] && [ -f "$VM_DIR/.osk" ]; then
    APPLESMC_OSK="$(tr -d '[:space:]' < "$VM_DIR/.osk")"
fi
if [ -z "${APPLESMC_OSK:-}" ]; then
    echo "APPLESMC_OSK is not set, and $VM_DIR/.osk does not exist." >&2
    echo "The guest will not boot without it." >&2
    exit 1
fi

cd "$VM_DIR"

args=(
  -enable-kvm -m "$RAM_MIB"
  # Skylake-Client, not Penryn. Penryn is the usual advice for AMD hosts, but
  # that advice predates Sequoia: the installer tolerates it and the installed
  # system does not, panicking into a boot loop that looks like a broken
  # install rather than a CPU model problem.
  #
  # The Intel vendor string is required regardless -- macOS does not run on a
  # CPU that reports AuthenticAMD.
  -cpu "Skylake-Client,-hle,-rtm,kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check"
  -machine q35
  -device qemu-xhci,id=xhci
  -device usb-kbd,bus=xhci.0 -device usb-tablet,bus=xhci.0
  -smp "$THREADS",cores="$CORES",sockets=1
  -device isa-applesmc,osk="$APPLESMC_OSK"
  -drive if=pflash,format=raw,readonly=on,file=OVMF_CODE_4M.fd
  -drive if=pflash,format=raw,file=OVMF_VARS-1920x1080.fd
  -smbios type=2
  -device ich9-ahci,id=sata
  -drive id=OpenCoreBoot,if=none,snapshot=on,format=qcow2,file=OpenCore/OpenCore.qcow2
  -device ide-hd,bus=sata.2,drive=OpenCoreBoot
  -drive id=MacHDD,if=none,file=mac_hdd_ng.img,format=qcow2
  -device ide-hd,bus=sata.4,drive=MacHDD
  # ssh: host $SSH_PORT -> guest 22, on loopback ONLY. Written as tcp::2222
  # this binds 0.0.0.0 and puts the guest's ssh on the LAN, which with a
  # simple password is not a door to leave open.
  -netdev user,id=net0,hostfwd=tcp:127.0.0.1:"$SSH_PORT"-:22
  -device virtio-net-pci,netdev=net0,id=net0,mac="$MAC"
  -device vmware-svga
  # No QEMU window: the VM must not depend on anyone being logged into the
  # desktop. Once Remote Login is on, nothing needs the display at all.
  -display none
  -vnc 127.0.0.1:"$VNC_DISPLAY"
  # The monitor is how the guest is managed without a display, including live
  # storage migration (drive_mirror) -- which is how this install was moved
  # between disks halfway through without restarting it.
  -monitor unix:"$MONITOR",server,nowait
)

# Only while installing. Leaving the installer attached afterwards gives the
# boot picker a second entry that is easy to select by accident.
if [ -f BaseSystem.img ] && [ -n "${WITH_INSTALLER:-}" ]; then
    args+=(-device ide-hd,bus=sata.3,drive=InstallMedia
           -drive id=InstallMedia,if=none,file=BaseSystem.img,format=raw)
fi

exec qemu-system-x86_64 "${args[@]}"
