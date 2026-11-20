#!/bin/sh
# Perlin density ladder R1-R6, hand-written SPIR-V through wgpu's SPIR-V
# passthrough. One command, one report: adapter, driver, the Vulkan float
# controls properties that decide whether the contract can be honoured, and
# the bit-mismatch count per rung for three module variants.
#
#   tools/perlin_spirv_probe.sh              # current $DISPLAY/$VK_ICD_FILENAMES
#   VK_ICD_FILENAMES=... tools/perlin_spirv_probe.sh
#
# Variants:
#   base              every float op decorated NoContraction
#   --float-controls  base + DenormPreserve + SignedZeroInfNanPreserve
#                     (SPV_KHR_float_controls2)
#   --allow-contraction  base with the NoContraction decorations removed
#                     (negative control: is the driver already exact?)
#
# Every rung must read 0/60,000 for the run to pass. Needs `spirv-as` and
# `vulkaninfo`; the probe itself needs `spirv-as` on PATH at runtime.
set -u

cd "$(dirname "$0")/.." || exit 1

echo "== host =="
uname -srm

echo
echo "== adapter =="
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | sed -n '/^Devices:/,$p' | grep -E \
        '^\s*(apiVersion|driverVersion|deviceType|deviceName|driverID|driverName|driverInfo|conformanceVersion)\s*='
else
    echo "vulkaninfo not found; adapter comes from the probe instead"
fi

echo
echo "== float controls =="
if command -v vulkaninfo >/dev/null 2>&1; then
    # First physical device only: the second block belongs to the next GPU.
    vulkaninfo --show-promoted-structs 2>/dev/null | awk '
        /VkPhysicalDeviceFloatControlsProperties:/ { inblock = 1; next }
        inblock && /^[A-Za-z_]+Features:|^VkPhysicalDevice.*Properties:/ { exit }
        inblock && /denormBehaviorIndependence|roundingModeIndependence|shader[A-Za-z0-9]*Float32/ {
            print
        }
    ' | sort -u
    vulkaninfo 2>/dev/null | grep -q 'VK_KHR_shader_float_controls2' \
        && echo "VK_KHR_shader_float_controls2 = present" \
        || echo "VK_KHR_shader_float_controls2 = absent"
else
    echo "unavailable (no vulkaninfo)"
fi

echo
echo "== build =="
cargo build -q --profile fast -p horizon-client-lib --example perlin_spirv_probe || exit 1
probe=target/fast/examples/perlin_spirv_probe

status=0
for mode in base --float-controls --allow-contraction; do
    echo
    echo "== $mode =="
    if [ "$mode" = base ]; then
        "$probe"
    else
        "$probe" "$mode"
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAILED ($rc)"
        status=$rc
    fi
done

echo
if [ $status -eq 0 ]; then
    echo "RESULT: all three variants bit-exact on every rung"
else
    echo "RESULT: at least one variant diverged (see tables above)"
fi
exit $status
