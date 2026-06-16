"""
read_ss.py - standalone reader for Xpatch .ss signature files.

Ported from the hand-transcribed MATLAB ssread.m / xpheaders.m in this project.
READ ONLY (no write support). Pulls exactly what GRIM needs: the complex
scattering data (VV/VH/HV/HH), the frequency axis, and the azimuth/elevation of
each signal.

How it works -- the .ss format is self-framing. Every "signal" record starts
with two big-endian int32s, nbytesb and nbytesd:

    record_size = nbytesb + nbytesd
    nbytesb     = byte offset from record start to header-D
    nbytesd     = header-D (408 bytes) + complex data block
                  => num_freqs = (nbytesd - 408) / 32
                     (32 bytes per freq = 4 pols x complex64)

That framing lets us reach header-D (az/el) and the data block WITHOUT parsing
the variable-length B / advanced / raytrace / materials blocks that ssread.m
walks through (their contents are unused here). The only table-derived read is
header-C (the frequency axis); we cross-check it by requiring header-C `maxfreq`
to equal the framing-derived num_freqs -- if those disagree, the header-C offset
(and thus the freq axis) is suspect, while az/el/data stay correct because they
are pinned by the framing ints, not by the field tables.

VERIFY the printed numbers against MATLAB `ssread` output before trusting this.
"""

import sys
import numpy as np

BE_I4 = np.dtype(">i4")   # big-endian int32  (MATLAB '*int',  'ieee-be')
BE_F4 = np.dtype(">f4")   # big-endian single (MATLAB '*float', 'ieee-be')

_TYPE_BYTES = {"int": 4, "float": 4, "char": 1, "": 1}

# --- field tables, ported verbatim from xpheaders.m -------------------------
# (type, name, count).  type "" / name "" means: skip `count` BYTES.

HDRA = [
    ("int", "nbytesb", 1), ("int", "nbytesd", 1),
    ("char", "bin_head_form", 1), ("char", "method", 1),
    ("char", "edge_diff", 1), ("char", "polar", 1),
    ("char", "x_version_num", 16), ("char", "hardware", 8),
    ("char", "host_machine", 16), ("char", "op_system", 16),
    ("char", "op_release", 8), ("char", "op_version", 8),
    ("char", "mem_version", 8), ("char", "num_tasks", 1),
    ("char", "simTitle", 256), ("char", "simDate", 3),
    ("char", "restart_Date", 3), ("char", "restart_count", 1),
    ("char", "ipoedge", 1), ("char", "iqmatrix", 1),
    ("char", "ibspsave", 1), ("char", "acadfct", 256),
]

HDRB_BYTES = 3 * 256  # acadedge, acadcurv, acadbsp -- 3 x char[256]

HDRC = [
    ("int","maxlay",1),("int","maxrstep",1),("int","maxchild",1),("int","maxson",1),
    ("int","maxram",1),("int","maxband",1),("int","maxpixx",1),("int","maxpixy",1),
    ("int","max114knot",1),("int","igui",1),("float","safenuss",1),("float","edgeblockwave",1),
    ("int","maxfiles",1),("int","maxaspects",1),("int","maxang",1),("int","maxfreqbin",1),
    ("int","maxbncpl",1),("int","maxcoat",1),("int","maxbulkcoat",1),("int","maxexpnuss",1),
    ("int","maxfreq",1),("int","maxedge",1),("int","maxfreqang",1),("int","maxramf",1),
    ("int","maxrangestep",1),("int","maxstackson",1),("int","iramtot",1),("int","maxnfct",1),
    ("int","maxnnod",1),("char","modelTitle",256),("float","model_roll_angle",1),("float","bmin",3),
    ("float","bmax",3),("int","mctot",1),("int","mcbadtot",1),("int","mabsorbtot",1),
    ("float","areatot",1),("int","itracetype",1),("int","iunit",1),("int","ifreq",1),
    ("float","freq1",1),("float","freq2",1),("int","nfreq",1),("int","inorange",1),
    ("float","range1",1),("float","range2",1),("int","nrange",1),("int","imono",1),
    ("float","rt071",1),("float","rt072",1),("int","nrt07",1),("float","rp071",1),
    ("float","rp072",1),("int","nrp07",1),("float","theob1",1),("float","theob2",1),
    ("int","ntheob",1),("float","phiob1",1),("float","phiob2",1),("int","nphiob",1),
    ("int","ioutformat",1),("int","iaddedge",1),("int","ipozbuff",1),("float","cellmax",1),
    ("float","blockangle",1),("int","irightnormal",1),("float","pixsize",1),("int","ipixout",1),
    ("int","maxvoxdepth",1),("int","maxvoxl",1),("int","maxbncin",1),("float","raywvel",1),
    ("float","raywvaz",1),("float","nscale",1),("int","icoatabsorb",1),("int","ipec",1),
    ("float","pecfudge",1),("float","delf9",1),("int","maxang_in",1),("int","num_advanced",1),
]

# header-D minimal form (the 'd' case): 280 skip, az/el/azobs/elobs, 112 skip
# 280 + 4*4 + 112 = 408  <- matches the magic number in ssread.m
HDRDMIN = [
    ("", "", 280),
    ("float", "azinc", 1), ("float", "elinc", 1),
    ("float", "azobs", 1), ("float", "elobs", 1),
    ("", "", 112),
]


def _table_bytes(table):
    return sum(_TYPE_BYTES[t] * n for t, _, n in table)


def _parse_table(buf, table):
    """Parse a packed big-endian record `buf` per `table`; return name -> value(s)."""
    out, off = {}, 0
    for typ, name, count in table:
        nbytes = _TYPE_BYTES[typ] * count
        chunk = buf[off:off + nbytes]
        if typ == "int":
            out[name] = np.frombuffer(chunk, BE_I4, count)
        elif typ == "float":
            out[name] = np.frombuffer(chunk, BE_F4, count)
        elif typ == "char":
            out[name] = chunk.tobytes()
        off += nbytes
    return out


def _i4(buf, off):
    return int(np.frombuffer(buf[off:off + 4], BE_I4)[0])


def read_ss(path, verbose=True):
    raw = np.fromfile(path, dtype=np.uint8)
    filesize = raw.size
    if filesize < 8:
        raise ValueError(f"{path}: too small to be a .ss file ({filesize} bytes)")

    size_a = _table_bytes(HDRA)        # = 615
    hdrc_off = size_a + HDRB_BYTES     # header-C offset within record 0

    # --- header C (frequency axis); read once from the first record ----------
    hdrc = _parse_table(raw[hdrc_off:hdrc_off + _table_bytes(HDRC)], HDRC)
    ifreq = int(hdrc["ifreq"][0])
    maxfreq = int(hdrc["maxfreq"][0])
    freq1 = float(hdrc["freq1"][0])
    freq2 = float(hdrc["freq2"][0])

    # --- walk records by framing --------------------------------------------
    az, el = [], []
    pol = {"vv": [], "vh": [], "hv": [], "hh": []}
    p, nsig, num_freqs_global = 0, 0, None
    while p + 8 <= filesize:
        nbytesb = _i4(raw, p)
        nbytesd = _i4(raw, p + 4)
        if nbytesb <= 0 or nbytesd <= 408:
            if verbose:
                print(f"  record {nsig}: framing not set (nbytesb={nbytesb}, "
                      f"nbytesd={nbytesd}); stopping")
            break
        num_freqs = (nbytesd - 408) // 32
        if num_freqs_global is None:
            num_freqs_global = num_freqs

        # header-D (minimal): az/el at p + nbytesb
        d = _parse_table(raw[p + nbytesb: p + nbytesb + 408], HDRDMIN)
        az.append(float(d["azinc"][0]))
        el.append(float(d["elinc"][0]))

        # complex data: 32*num_freqs bytes at p + nbytesb + 408
        dstart = p + nbytesb + 408
        nbytes = 32 * num_freqs
        if dstart + nbytes > filesize:
            if verbose:
                print(f"  record {nsig}: data block truncated; stopping")
            break
        chunk = np.frombuffer(raw[dstart:dstart + nbytes], BE_F4, 8 * num_freqs)
        c = chunk[0::2] + 1j * chunk[1::2]      # num_freqs*4 complex
        pol["vv"].append(c[0::4]); pol["vh"].append(c[1::4])
        pol["hv"].append(c[2::4]); pol["hh"].append(c[3::4])

        nsig += 1
        p += nbytesb + nbytesd

    if nsig == 0:
        raise ValueError(f"{path}: no readable signal records")

    # --- frequency axis ------------------------------------------------------
    if ifreq == 2:
        # explicit freqs: maxfreq float32 immediately before header-D of record 0
        nbytesb0 = _i4(raw, 0)
        fstart = nbytesb0 - 4 * maxfreq
        freqdata = np.frombuffer(raw[fstart:fstart + 4 * maxfreq], BE_F4, maxfreq).copy()
    else:
        freqdata = np.linspace(freq1, freq2, maxfreq)

    match = (num_freqs_global == maxfreq)
    result = {
        "az": np.asarray(az), "el": np.asarray(el),
        "freq": freqdata, "num_freqs": num_freqs_global,
        "maxfreq": maxfreq, "ifreq": ifreq,
        "vv": np.asarray(pol["vv"]), "vh": np.asarray(pol["vh"]),   # (nsig, num_freqs)
        "hv": np.asarray(pol["hv"]), "hh": np.asarray(pol["hh"]),
        "header_c": hdrc, "freq_axis_ok": match,
    }

    if verbose:
        print(f"  signals           : {nsig}")
        print(f"  num_freqs (framing): {num_freqs_global}    maxfreq (header C): {maxfreq}    match: {match}")
        if not match:
            print("  !! mismatch -> header-C offset/tables look off; FREQ AXIS SUSPECT")
            print("     (az/el/data are framing-pinned and still trustworthy)")
        print(f"  ifreq             : {ifreq}   freq1={freq1:.6g}  freq2={freq2:.6g}")
        print(f"  az range          : {min(az):.4f} .. {max(az):.4f}")
        print(f"  el range          : {min(el):.4f} .. {max(el):.4f}")
        print(f"  freq[:3]          : {np.round(freqdata[:3], 6)}")
        print(f"  vv[sig0][:2]      : {result['vv'][0][:2]}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python read_ss.py FILE.ss")
        sys.exit(1)
    print(f"reading {sys.argv[1]}")
    read_ss(sys.argv[1])
