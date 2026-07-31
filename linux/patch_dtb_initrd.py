#!/usr/bin/env python3
"""Add /chosen/linux,initrd-start|end to a flattened device-tree blob.

The JTAG boot flow (vitis_linux/scripts/boot_linux_jtag.tcl) hand-boots the
kernel with no U-Boot (r0=0 r1=~0 r2=dtb pc=zImage), so the kernel locates the
initramfs already loaded in DDR via the /chosen/linux,initrd-start and
/chosen/linux,initrd-end properties (u32 physical addresses; the raw gzipped
cpio in initramfs.cpio.gz is loaded there with `dow -data`).

This rewrites the blob to a new file with those two properties added to
/chosen (replacing any existing ones). Called from linux/build_all.sh after the
buildroot step, which knows the actual initramfs size.

Usage:
  python3 patch_dtb_initrd.py <in.dtb> <initrd_start_hex> <initrd_size> <out.dtb>

Output layout (FDT v16/17): header + mem_rsvmap (unchanged) + rebuilt structure
block (only /chosen gains the two props) + rebuilt strings block. Verified on
Windows against the real devicetree.dtb (canonical prop diff: only the two new
props differ).
"""

import struct
import sys

FDT_BEGIN_NODE = 1
FDT_END_NODE = 2
FDT_PROP = 3
FDT_NOP = 4
FDT_END = 9
MAGIC = 0xD00DFEED


def main():
    if len(sys.argv) != 5:
        print("usage: patch_dtb_initrd.py <in.dtb> <initrd_start_hex> <initrd_size> <out.dtb>")
        return 2
    in_path, start_hex, size_s, out_path = sys.argv[1:5]
    start = int(start_hex, 0)
    size = int(size_s, 0)

    data = bytearray(open(in_path, "rb").read())
    (magic, totalsize, off_struct, off_strings, off_rsv,
     boot_cpuid, version, last_comp, size_strings, size_struct) = \
        struct.unpack_from(">10I", data, 0)
    if magic != MAGIC:
        raise SystemExit("not an FDT (magic=0x%08x)" % magic)
    end_struct = off_struct + size_struct
    if off_strings + size_strings > len(data) or end_struct > off_strings:
        raise SystemExit("truncated/invalid dtb")

    strings_orig = bytes(data[off_strings:off_strings + size_strings])

    def str_orig(nameoff):
        end = strings_orig.index(b"\x00", nameoff)
        return strings_orig[nameoff:end].decode("ascii")

    strings_new = bytearray()
    str_off = {}

    def add_string(name):
        nonlocal strings_new
        if name in str_off:
            return str_off[name]
        str_off[name] = len(strings_new)
        strings_new += name.encode("ascii") + b"\x00"
        return str_off[name]

    initrd_start_off = add_string("linux,initrd-start")
    initrd_end_off = add_string("linux,initrd-end")

    struct_new = bytearray()
    pos = off_struct
    depth = -1
    in_chosen = False
    while pos < end_struct:
        tok = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        if tok == FDT_BEGIN_NODE:
            end = data.index(b"\x00", pos)
            name = data[pos:end].decode("ascii")
            if depth == 0 and name == "chosen":
                in_chosen = True
            depth += 1
            struct_new += struct.pack(">I", tok)
            struct_new += data[pos:end + 1]
            while len(struct_new) % 4:
                struct_new.append(0)
            pos = (end + 1 + 3) & ~3
        elif tok == FDT_END_NODE:
            if in_chosen:
                for pname, val in (("linux,initrd-start", start),
                                   ("linux,initrd-end", start + size)):
                    struct_new += struct.pack(">III", FDT_PROP, 4, str_off[pname])
                    struct_new += struct.pack(">I", val)
            struct_new += struct.pack(">I", tok)
            depth -= 1
            in_chosen = False
        elif tok == FDT_PROP:
            plen, nameoff = struct.unpack_from(">II", data, pos)
            data_off = pos + 8
            name = str_orig(nameoff)
            skip = in_chosen and name in ("linux,initrd-start", "linux,initrd-end")
            if not skip:
                struct_new += struct.pack(">III", FDT_PROP, plen, add_string(name))
                struct_new += data[data_off:data_off + plen]
                while len(struct_new) % 4:
                    struct_new.append(0)
            pos = data_off + plen
            pos = (pos + 3) & ~3
        elif tok == FDT_NOP:
            pass
        elif tok == FDT_END:
            struct_new += struct.pack(">I", tok)
            pos = end_struct
        else:
            raise SystemExit("bad token 0x%08x at struct offset %d" % (tok, pos - 4))

    while len(struct_new) % 8:
        struct_new.append(0)
    while len(strings_new) % 8:
        strings_new.append(0)

    off_strings_new = off_struct + len(struct_new)
    totalsize_new = (off_strings_new + len(strings_new) + 7) & ~7

    header = struct.pack(">10I", MAGIC, totalsize_new, off_struct, off_strings_new,
                         off_rsv, boot_cpuid, version, last_comp,
                         len(strings_new), len(struct_new))
    blob = header + data[off_rsv:off_struct] + struct_new + strings_new
    with open(out_path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes): /chosen/linux,initrd-start=0x%x "
          "linux,initrd-end=0x%x" % (out_path, len(blob), start, start + size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
