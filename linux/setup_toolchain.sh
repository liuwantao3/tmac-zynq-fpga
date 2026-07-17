#!/bin/bash
# Set up clang-based arm-linux-gnueabihf cross-compiler toolchain
# Uses macOS clang + brew arm-linux-gnueabihf-binutils — no downloads needed.
# Usage: bash linux/setup_toolchain.sh

set -euo pipefail
TOOLS="/tmp/arm-toolchain/bin"
mkdir -p "$TOOLS"

# ── Check brew binutils ──
for t in ld as objcopy objdump nm ar ranlib strip readelf; do
    which "arm-linux-gnueabihf-$t" >/dev/null 2>&1 || {
        echo "Missing: arm-linux-gnueabihf-$t — install with:"
        echo "  brew install arm-linux-gnueabihf-binutils"
        exit 1
    }
done

# ── clang wrapper for arm-linux-gnueabihf-gcc ──
cat > "$TOOLS/arm-linux-gnueabihf-gcc" << 'CLANGWRAP'
#!/bin/zsh
args=()
is_mthumb=0
for a in "$@"; do
    case "$a" in
        -mthumb)          is_mthumb=1 ;;
        -mapcs-frame|-mgeneral-regs-only|-mno-unaligned-access|-mabi=apcs-gnu) ;;
        -fno-ipa-sra|-fno-section-anchors|-femit-struct-debug-baseonly) ;;
        -fconserve-stack|-fno-inline-functions-called-once|-fno-var-tracking*) ;;
        -fno-unit-at-a-time|-fpartial-inlining|-fno-peephole*) ;;
        -fno-jump-tables|-fno-reorder-blocks|-fno-optimize-sibling-calls) ;;
        -fcf-protection=none|-fno-shrink-wrap|-fno-merge-all-constants) ;;
        -freorder-blocks-algorithm=*|-flive-patching=*|-fstack-usage) ;;
        -finstrument-functions|-finline-limit=*|-fno-stack-check|-fno-PIE) ;;
        -mabi=aapcs-linux|-mfix-cortex-*|-mno-fix-cortex-*|-mno-thumb-interwork) ;;
        -mthumb-interwork|-mno-bti-at-return-twice|-mslow-flash-data) ;;
        *) args+=("$a") ;;
    esac
done
[[ $is_mthumb -eq 0 ]] && args+=("-marm")
exec clang --target=armv7-linux-gnueabihf -march=armv7-a -mfloat-abi=hard -mfpu=neon -integrated-as "${args[@]}"
CLANGWRAP
chmod +x "$TOOLS/arm-linux-gnueabihf-gcc"

# ── Symlinks ──
for tool in g++ cpp gcc-ar gcc-nm gcc-ranlib; do
    ln -sf arm-linux-gnueabihf-gcc "$TOOLS/arm-linux-gnueabihf-$tool"
done

# Symlink brew binutils
for tool in addr2line ar as c++filt elfedit gprof ld ld.bfd nm objcopy objdump ranlib readelf size strings strip; do
    src=$(which "arm-linux-gnueabihf-$tool" 2>/dev/null)
    [[ -n "$src" ]] && ln -sf "$src" "$TOOLS/arm-linux-gnueabihf-$tool"
done

echo "Toolchain ready at $TOOLS"
echo "Add to PATH:  export PATH=\"$TOOLS:\$PATH\""
echo ""
"$TOOLS/arm-linux-gnueabihf-gcc" --version 2>/dev/null | head -1
echo ""
echo "Try: echo 'int main(void){return 0;}' | arm-linux-gnueabihf-gcc -x c - -c -o /dev/null"
