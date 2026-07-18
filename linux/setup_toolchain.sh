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

# ── Create minimal elf.h for macOS (no system-level ELF headers) ──
cat > "$TOOLS/elf.h" << 'ELFEOF'
/* Minimal elf.h for macOS — provides types needed by Linux kernel build tools */
#ifndef _MACOS_ELF_H
#define _MACOS_ELF_H
#include <stdint.h>
typedef uint16_t Elf32_Half; typedef uint32_t Elf32_Word; typedef int32_t Elf32_Sword;
typedef uint32_t Elf32_Addr; typedef uint32_t Elf32_Off;
#define EI_NIDENT 16
typedef struct { unsigned char e_ident[EI_NIDENT]; Elf32_Half e_type, e_machine; Elf32_Word e_version; Elf32_Addr e_entry; Elf32_Off e_phoff, e_shoff; Elf32_Word e_flags; Elf32_Half e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx; } Elf32_Ehdr;
typedef struct { Elf32_Word sh_name, sh_type, sh_flags; Elf32_Addr sh_addr; Elf32_Off sh_offset; Elf32_Word sh_size, sh_link, sh_info, sh_addralign, sh_entsize; } Elf32_Shdr;
typedef struct { Elf32_Word st_name; Elf32_Addr st_value; Elf32_Word st_size; unsigned char st_info, st_other; Elf32_Half st_shndx; } Elf32_Sym;
typedef uint64_t Elf64_Addr, Elf64_Off, Elf64_Xword; typedef int64_t Elf64_Sxword;
typedef uint16_t Elf64_Half; typedef uint32_t Elf64_Word; typedef int32_t Elf64_Sword;
typedef struct { unsigned char e_ident[EI_NIDENT]; Elf64_Half e_type, e_machine; Elf64_Word e_version; Elf64_Addr e_entry; Elf64_Off e_phoff, e_shoff; Elf64_Word e_flags; Elf64_Half e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx; } Elf64_Ehdr;
typedef struct { Elf64_Word sh_name, sh_type; Elf64_Xword sh_flags; Elf64_Addr sh_addr; Elf64_Off sh_offset; Elf64_Xword sh_size; Elf64_Word sh_link, sh_info; Elf64_Xword sh_addralign, sh_entsize; } Elf64_Shdr;
typedef struct { Elf64_Word st_name; unsigned char st_info, st_other; Elf64_Half st_shndx; Elf64_Addr st_value; Elf64_Xword st_size; } Elf64_Sym;
#define ELFMAG0 0x7f; #define ELFMAG1 'E'; #define ELFMAG2 'L'; #define ELFMAG3 'F'; #define ELFMAG "\177ELF"; #define SELFMAG 4
#define EI_MAG0 0; #define EI_MAG1 1; #define EI_MAG2 2; #define EI_MAG3 3; #define EI_DATA 5
#define ELFDATA2LSB 1; #define ELFDATA2MSB 2
#define ELFCLASS32 1; #define ELFCLASS64 2
#define SHT_SYMTAB 2; #define SHT_SYMTAB_SHNDX 18
#define ELF32_ST_TYPE(v) ((unsigned char)(v))
#define ELF32_ST_INFO(b,t) (((b)<<4)+((t)&0xf))
#define ELF64_ST_TYPE(v) ELF32_ST_TYPE(v)
#define ELF64_ST_INFO(b,t) ELF32_ST_INFO(b,t)
#define STT_OBJECT 1; #define STT_FUNC 2; #define STT_NOTYPE 0
#define ET_EXEC 2; #define ET_DYN 3
#endif
ELFEOF
echo "  elf.h created for kernel host builds"
echo ""
"$TOOLS/arm-linux-gnueabihf-gcc" --version 2>/dev/null | head -1
echo ""
echo "Try: echo 'int main(void){return 0;}' | arm-linux-gnueabihf-gcc -x c - -c -o /dev/null"
