#!/usr/bin/env python3
"""
Measure the relative speed (nps) of two engine builds using their `bench` command.

Based on the original speedup.py by Koivisto dev Luecx, with corrected statistics.

The engines are run alternately, one bench each per round, and the speed ratio is
computed *per round*. Pairing the runs in time cancels slow drift (CPU warm-up,
thermal throttling, background load), which otherwise dominates the comparison.

Reported uncertainty is the confidence interval of the *mean* ratio (sigma/sqrt(n)),
not the spread of a single measurement -- the original script printed 3*sigma of the
raw samples, which never shrinks with more data and made real differences look
insignificant forever.

Rounds disturbed by other activity on the machine are detected against the median and
rejected, and rows are colored by how well the result is established: dim while still
unresolved, green once significant, yellow/red for disturbed rounds and slowdowns.

Usage:
    python speedup.py                       # the two executables next to this script
    python speedup.py new.exe old.exe       # explicit, first one is the new build
    python speedup.py --engine-dir path/to/engines --depth 14

Press Ctrl+C at any time to stop and print the final summary.
"""

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# terminal colors
# ---------------------------------------------------------------------------

RESET, DIM, GREEN, BRIGHT_GREEN, YELLOW, RED = (
    "\033[0m", "\033[2m", "\033[32m", "\033[92m", "\033[33m", "\033[91m")

_use_color = False


def init_color(enabled):
    """Enable ANSI output, turning on virtual terminal processing on Windows."""
    global _use_color
    if not enabled or os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        _use_color = False
        return
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # 7 = STD_OUTPUT_HANDLE, 0x4 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return
            kernel32.SetConsoleMode(handle, mode.value | 0x4)
        except Exception:
            return
    _use_color = True


def paint(text, color):
    return f"{color}{text}{RESET}" if _use_color and color else text


def confidence_color(mean, margin):
    """Color a row by how far the confidence interval sits from zero.

    Driven by the interval rather than the raw t statistic: with only a handful of
    rounds even a large t is not significant, and a row must never look established
    while its own CI still includes 0.
    """
    if margin <= 0.0:
        return DIM
    significance = mean / margin  # >= 1 means the interval excludes zero
    if significance >= 2.0:
        return BRIGHT_GREEN
    if significance >= 1.0:
        return GREEN
    if significance <= -1.0:
        return RED
    return DIM

# ---------------------------------------------------------------------------
# bench running
# ---------------------------------------------------------------------------

# "<nodes> nodes <nps> nps" as printed by the engine's bench command
NODES_RE = re.compile(r"(\d+)\s*nodes")
NPS_RE = re.compile(r"(\d+)\s*nps")

HIGH_PRIORITY_CLASS = 0x00000080  # Windows CREATE_* flag


def _popen_priority_kwargs(high_priority):
    """Platform-specific arguments that raise the bench process priority."""
    if not high_priority:
        return {}
    if sys.platform == "win32":
        return {"creationflags": HIGH_PRIORITY_CLASS}

    def raise_priority():
        try:
            os.nice(-5)
        except OSError:
            pass  # no privileges, run at default priority

    return {"preexec_fn": raise_priority}


def run_bench(engine, depth, high_priority, timeout):
    """Run one bench and return (nodes, nps). Raises RuntimeError if unparseable."""
    argv = [engine, f"bench {depth}" if depth else "bench"]
    proc = subprocess.Popen(
        argv,
        cwd=os.path.dirname(os.path.abspath(engine)) or None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        **_popen_priority_kwargs(high_priority),
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"{os.path.basename(engine)}: bench timed out after {timeout}s")
    except KeyboardInterrupt:
        proc.kill()
        raise

    # the summary line is at the end, so scan backwards
    for line in reversed(output.splitlines()):
        nps = NPS_RE.search(line)
        if nps:
            nodes = NODES_RE.search(line)
            return (int(nodes.group(1)) if nodes else None, int(nps.group(1)))

    raise RuntimeError(
        f"{os.path.basename(engine)}: no 'nps' found in bench output:\n{output.strip()[-500:]}"
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

# Two-sided Student-t critical values for small sample sizes. For df > 30 a
# Cornish-Fisher expansion around the normal quantile is accurate to <0.1%.
_T_TABLE = {
    90: [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
         1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
         1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697],
    95: [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042],
    99: [63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169,
         3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878, 2.861, 2.845,
         2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771, 2.763, 2.756, 2.750],
}
_Z = {90: 1.644854, 95: 1.959964, 99: 2.575829}


def t_critical(df, confidence):
    if df < 1:
        return float("inf")
    table = _T_TABLE[confidence]
    if df <= len(table):
        return table[df - 1]
    z = _Z[confidence]
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


class Samples:
    """Running mean / sample standard deviation (Welford)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def sigma(self):
        """Spread of a single measurement."""
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def stderr(self):
        """Uncertainty of the mean -- this is what shrinks as samples accumulate."""
        return self.sigma / math.sqrt(self.n) if self.n > 1 else 0.0

    def margin(self, confidence):
        """Half-width of the confidence interval of the mean."""
        return t_critical(self.n - 1, confidence) * self.stderr if self.n > 1 else 0.0


# A single disturbed round (a browser starting, a build kicking off) can be tens of
# percent off and biases the mean far more than the effect being measured, so such
# rounds are detected against the median and dropped.
MIN_OUTLIER_SAMPLES = 8   # need a stable median before anything can be called an outlier
SUSPICIOUS_DEVIATION = 3.0
MIN_ROBUST_SIGMA = 0.002  # floor, so an unusually tight early cluster cannot make
                          # ordinary rounds look like outliers


def deviation(x, samples):
    """How far x lies from the sample median, in robust sigmas (median absolute deviation).

    MAD is used rather than the standard deviation because the standard deviation is
    itself inflated by the very outlier being tested.
    """
    if len(samples) < MIN_OUTLIER_SAMPLES:
        return 0.0
    med = statistics.median(samples)
    mad = statistics.median([abs(v - med) for v in samples])
    return abs(x - med) / max(1.4826 * mad, MIN_ROBUST_SIGMA)


# ---------------------------------------------------------------------------
# engine discovery
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_engines(engine_dir):
    found = []
    for name in sorted(os.listdir(engine_dir)):
        path = os.path.join(engine_dir, name)
        if not os.path.isfile(path) or name.lower().endswith((".py", ".pnn", ".txt", ".md")):
            continue
        if sys.platform == "win32":
            if name.lower().endswith(".exe"):
                found.append(path)
        elif os.access(path, os.X_OK):
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare bench speed of two engine builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Press Ctrl+C to stop and print the final summary.",
    )
    p.add_argument("engines", nargs="*", metavar="ENGINE",
                   help="two engine executables: NEW first, then OLD (baseline). "
                        "If omitted, the engine directory is scanned instead.")
    p.add_argument("--engine-dir", metavar="DIR", default=SCRIPT_DIR,
                   help="directory to scan for engines when none are given explicitly "
                        "(default: the directory containing this script)")
    p.add_argument("--depth", type=int, default=12, help="bench depth (default: 12)")
    p.add_argument("--warmup", type=int, default=1, metavar="N",
                   help="discard the first N rounds so clocks and caches settle (default: 1)")
    p.add_argument("--confidence", type=int, choices=(90, 95, 99), default=95,
                   help="confidence level for the reported interval (default: 95)")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-bench timeout in seconds (default: 600)")
    p.add_argument("--no-priority", action="store_true",
                   help="do not raise the bench process priority")
    p.add_argument("--outlier-threshold", type=float, default=5.0, metavar="MAD",
                   help="reject rounds deviating more than this many robust sigmas from the "
                        "median ratio; 0 disables rejection (default: 5)")
    p.add_argument("--no-color", action="store_true", help="disable colored output")
    args = p.parse_args()

    if len(args.engines) == 1:
        p.error("need two engines to compare, got one")
    elif len(args.engines) > 2:
        p.error(f"need exactly two engines to compare, got {len(args.engines)}")
    elif not args.engines:
        if not os.path.isdir(args.engine_dir):
            p.error(f"engine directory does not exist: {args.engine_dir}")
        found = find_engines(args.engine_dir)
        if len(found) != 2:
            listing = "\n  ".join(os.path.basename(f) for f in found) or "(none)"
            p.error(f"expected exactly 2 executables in {args.engine_dir}, found "
                    f"{len(found)}:\n  {listing}\nPass the two engines explicitly: "
                    f"speedup.py NEW OLD")
        args.engines = found
        # listdir order is arbitrary, so the new/old assignment is a guess
        print(f"note: no engines given, using the two executables found in {args.engine_dir}")

    for path in args.engines:
        if not os.path.isfile(path):
            p.error(f"engine not found: {path}")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.outlier_threshold < 0:
        p.error("--outlier-threshold must be >= 0")
    return args


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    args = parse_args()
    init_color(not args.no_color)
    new_engine, old_engine = args.engines
    names = [os.path.basename(e) for e in args.engines]
    conf = args.confidence

    print(f"new   : {new_engine}")
    print(f"old   : {old_engine}")
    print(f"config: depth {args.depth}, warmup {args.warmup} round(s), "
          f"{'high' if not args.no_priority else 'default'} priority, {conf}% CI, "
          f"outlier cutoff {args.outlier_threshold or 'off'}")
    print()
    print(f"{'round':>5} {names[0][:16]:>18} {names[1][:16]:>18} "
          f"{'speedup':>9} {'CI':>11} {'noise/run':>10} {'t':>7}")
    print("-" * 92)

    new_nps, old_nps, ratio = Samples(), Samples(), Samples()
    ratios = []           # accepted ratio samples, kept for the robust median
    rejected = []         # (round, ratio, deviation) of rounds thrown out
    suspicious = 0
    nodes_seen = {}
    node_mismatch_reported = False
    start = time.time()
    rnd = 0

    try:
        while True:
            rnd += 1
            try:
                n_nodes, n_nps = run_bench(new_engine, args.depth, not args.no_priority, args.timeout)
                o_nodes, o_nps = run_bench(old_engine, args.depth, not args.no_priority, args.timeout)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

            # A bench that changes node count is a functional change; comparing nps
            # across different node counts is not a clean speedup measurement.
            for name, nodes in ((names[0], n_nodes), (names[1], o_nodes)):
                if nodes is not None:
                    nodes_seen.setdefault(name, nodes)
            if not node_mismatch_reported and len(set(nodes_seen.values())) > 1:
                node_mismatch_reported = True
                detail = ", ".join(f"{k}={v}" for k, v in nodes_seen.items())
                print(f"WARNING: bench node counts differ ({detail}). This is a functional "
                      f"change, not a pure speedup -- nps is not directly comparable.",
                      file=sys.stderr)

            if rnd <= args.warmup:
                print(paint(f"{rnd:>5} {n_nps:>18,} {o_nps:>18,} {'(warmup)':>9}", DIM))
                continue

            r = n_nps / o_nps - 1.0
            dev = deviation(r, ratios)

            if args.outlier_threshold and dev > args.outlier_threshold:
                # Show this round's own numbers, not the running means -- they reveal
                # which of the two engines was disturbed.
                rejected.append((rnd, r, dev))
                print(paint(f"{rnd:>5} {n_nps:>18,} {o_nps:>18,} {100 * r:>8.3f}% "
                            f"   rejected outlier ({dev:.1f} MAD from median)", RED))
                continue

            new_nps.add(n_nps)
            old_nps.add(o_nps)
            ratio.add(r)
            ratios.append(r)

            margin = ratio.margin(conf)
            t_stat = ratio.mean / ratio.stderr if ratio.stderr > 0 else 0.0
            line = (f"{rnd:>5} {new_nps.mean:>18,.0f} {old_nps.mean:>18,.0f} "
                    f"{100 * ratio.mean:>8.3f}% {100 * margin:>10.3f}% "
                    f"{100 * ratio.sigma:>9.3f}% {t_stat:>7.2f}")
            if dev > SUSPICIOUS_DEVIATION:
                suspicious += 1
                print(paint(f"{line}   disturbed round? (own ratio {100 * r:+.2f}%, "
                            f"{dev:.1f} MAD, kept)", YELLOW))
            else:
                print(paint(line, confidence_color(ratio.mean, margin)))

    except KeyboardInterrupt:
        print()

    counted = ratio.n
    if counted < 2:
        print("Not enough measured rounds for a result.")
        return 1

    margin = ratio.margin(conf)
    lo, hi = 100 * (ratio.mean - margin), 100 * (ratio.mean + margin)
    print(f"Summary after {counted} counted round(s) "
          f"({args.warmup} warmup discarded, {format_duration(time.time() - start)} elapsed)")
    print(f"  {names[0]:<24} {new_nps.mean:>14,.0f} nps  "
          f"(+/- {100 * new_nps.sigma / new_nps.mean:.2f}% per run)")
    print(f"  {names[1]:<24} {old_nps.mean:>14,.0f} nps  "
          f"(+/- {100 * old_nps.sigma / old_nps.mean:.2f}% per run)")
    print()
    print(paint(f"  speedup     {100 * ratio.mean:+.3f}%   "
                f"{conf}% CI [{lo:+.3f}%, {hi:+.3f}%]", confidence_color(ratio.mean, margin)))
    print(f"  median      {100 * statistics.median(ratios):+.3f}%   "
          f"(robust cross-check; a large gap to the mean means the run was disturbed)")
    print(f"  run-to-run noise  {100 * ratio.sigma:.3f}% (1 sigma of a single round; "
          f"does not shrink with more rounds)")

    if lo > 0:
        verdict = paint("speedup confirmed", BRIGHT_GREEN)
    elif hi < 0:
        verdict = paint("SLOWDOWN confirmed", RED)
    else:
        needed = math.ceil(counted * (margin / abs(ratio.mean)) ** 2) if ratio.mean else 0
        verdict = "inconclusive: the interval includes 0"
        if needed and needed > counted:
            verdict += f" (~{needed} rounds needed to resolve an effect this size)"
        verdict = paint(verdict, YELLOW)
    print(f"  verdict     {verdict}")

    if rejected:
        worst = max(rejected, key=lambda e: e[2])
        print(paint(f"  {len(rejected)} round(s) rejected as outliers "
                    f"(worst: round {worst[0]}, {100 * worst[1]:+.2f}%). "
                    f"Was the machine in use?", YELLOW))
    if suspicious:
        print(paint(f"  {suspicious} round(s) kept but flagged as disturbed.", YELLOW))
    if node_mismatch_reported:
        print(paint("  NOTE: node counts differed between engines -- treat the above "
                    "with care.", RED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
