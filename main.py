"""Forecast a game launch day's Twitch viewers, hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Run as `python3 main.py` (Python 3.13, standard library only, no network). For every case file
<id>.in in $MAKEFUN_INPUT_DIR (else /input) it writes <id>.out to $MAKEFUN_OUTPUT_DIR (else
/output): {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]} for h in 2..24]} — five
quantiles of that hour's concurrent viewers in the game's Twitch category.

The model, in one paragraph. A launch day is a wave that starts rolling at midnight local time
(New Zealand first), so by the end of hour 1 (01:00 UTC) the day's shape is already constrained:
hour 1 catches the Asia/Pacific and European midnight wave, dips as Europe sleeps, then builds
again into the American evening. Every comparable launch in the case is reduced to the ratio of
each of its release-day hours to its own hour 1, and for each target hour a weighted regression
of the log ratio on two launch features is fit across the comparables: the log of the launch's
hour-1 size, and the log "trough depth" (hour 1 divided by the previous day's peak — how much of
the wave had already arrived before the day began). The case's median for the hour is its own
hour-1 print times the regression's prediction at the case's features (clamped to the pool's
observed range, so an out-of-scale launch is not extrapolated), and its spread is the regression's
residual spread, floored and scaled by 0.70 — the scale that left the 5%-95% band covering about
93% of the practice hours. Comparables are weighted by kind: early-access-contaminated launches
x0.25, weak comparables x0.5, launches that did not roll at midnight local x0.10.

Defence in depth: any failure in the model path (too few usable comparables, a singular fit, a
malformed field) falls back per case to the persistence forecast — every hour at hour 1's print
with a spread widening like sqrt(hour - 1) — so the program always writes a valid forecast file
and never crashes. On the kit's four practice launches the model scores a mean pinball loss of
0.1496 on the log scale against 0.6029 for the flat "every quantile equals hour 1" forecast and
0.3871 for the persistence baseline (see BUILD_NOTES.md).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]
W_EA = 0.25        # weight of an early-access-contaminated comparable
W_WEAK = 0.5       # weight of a "weak" comparable
W_NONROLL = 0.10   # weight of a comparable that did not unlock rolling at midnight local
SD_MULT = 0.70     # spread scale: residual sd is shrunk to this on the practice evidence
SD_FLOOR = 0.20    # never forecast a tighter log spread than this
MIN_POOL = 6       # below this many usable comparables the model abstains to the fallback


def parse_ts(value):
    """Parse an ISO 8601 UTC timestamp; raises on anything else."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def day_points(day_start, hourly):
    """Map hour (1..24) -> viewers for the points that fall on day_start's UTC date."""
    day = str(day_start)[:10]
    start = parse_ts(day_start)
    pts = {}
    for point in hourly or []:
        try:
            stamp = point.get("from") or point.get("at")
            if str(stamp)[:10] == day and point.get("viewers") is not None:
                hour = int((parse_ts(stamp) - start).total_seconds() // 3600) + 1
                if 1 <= hour <= 24:
                    pts[hour] = float(point["viewers"])
        except Exception:
            continue
    return pts


def prev_day_peak(day_start, hourly):
    """The largest viewers value in the 24 hours before day_start, or None."""
    try:
        start = parse_ts(day_start)
    except Exception:
        return None
    low = start - timedelta(hours=24)
    best = None
    for point in hourly or []:
        try:
            stamp = point.get("from") or point.get("at")
            when = parse_ts(stamp)
            viewers = point.get("viewers")
            if viewers is not None and low <= when < start:
                if best is None or float(viewers) > best:
                    best = float(viewers)
        except Exception:
            continue
    return best


def tier_of(unlock_text):
    """The comparable's tier as its unlock note declares it ('Tier: comparable', ...)."""
    match = re.search(r"Tier:\s*([a-z\-]+)", unlock_text or "")
    return match.group(1) if match else "?"


def is_rolling(unlock_text):
    """True when the unlock note describes a rolling midnight-local unlock."""
    text = (unlock_text or "").lower()
    return "midnight local" in text or "rolling" in text


def comparable_features(comp):
    """Reduce one comparable launch to its hour ratios and features, or None if unusable."""
    try:
        pts = day_points(comp.get("day_start"), comp.get("hourly"))
        first = pts.get(1)
        if not first or first <= 0:
            return None
        tier = tier_of(comp.get("unlock"))
        if tier == "target-history":
            return None  # the target game's own pre-release history is not a launch
        peak = prev_day_peak(comp.get("day_start"), comp.get("hourly"))
        weight = 1.0
        if tier == "early-access-contaminated":
            weight *= W_EA
        if tier == "weak":
            weight *= W_WEAK
        if not is_rolling(comp.get("unlock")):
            weight *= W_NONROLL
        if weight <= 1e-9:
            return None
        return {
            "h1": first,
            "weight": weight,
            "trough": (first / peak) if peak else None,
            "ratios": {h: pts[h] / first for h in pts if h >= 2},
        }
    except Exception:
        return None


def solve2(s11, s12, s22, y1, y2):
    """Solve a symmetric 2x2 system; return (0, 0) when it is (near-)singular."""
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        return 0.0, 0.0
    return (s22 * y1 - s12 * y2) / det, (s11 * y2 - s12 * y1) / det


def hour_forecast(pool, log_p1, log_trough, hour, zs):
    """The five quantiles of one hour's ratio, as log offsets from the case's hour 1."""
    data = []
    for feat in pool:
        ratio = feat["ratios"].get(hour)
        if ratio is None or ratio <= 0:
            continue
        x1 = math.log(feat["h1"]) - log_p1
        if log_trough is not None and feat["trough"]:
            x2 = math.log(feat["trough"]) - log_trough
        else:
            x2 = 0.0
        data.append((math.log(ratio), x1, x2, feat["weight"]))
    if len(data) < MIN_POOL:
        return None
    total_w = sum(d[3] for d in data)
    mean_y = sum(d[3] * d[0] for d in data) / total_w
    mean_1 = sum(d[3] * d[1] for d in data) / total_w
    mean_2 = sum(d[3] * d[2] for d in data) / total_w
    s11 = sum(d[3] * (d[1] - mean_1) ** 2 for d in data) / total_w + 1e-9
    s22 = sum(d[3] * (d[2] - mean_2) ** 2 for d in data) / total_w + 1e-9
    s12 = sum(d[3] * (d[1] - mean_1) * (d[2] - mean_2) for d in data) / total_w
    s1y = sum(d[3] * (d[1] - mean_1) * (d[0] - mean_y) for d in data) / total_w
    s2y = sum(d[3] * (d[2] - mean_2) * (d[0] - mean_y) for d in data) / total_w
    b1, b2 = solve2(s11, s12, s22, s1y, s2y)
    mu = mean_y - b1 * mean_1 - b2 * mean_2
    var = sum(
        d[3] * (d[0] - (mean_y + b1 * (d[1] - mean_1) + b2 * (d[2] - mean_2))) ** 2
        for d in data
    ) / total_w
    sd = max(SD_FLOOR, math.sqrt(max(var, 0.0))) * SD_MULT
    return [mu + z * sd for z in zs]


def model_forecast(case, p1, hours, zs):
    """The regression-pool forecast: {hour: [quantiles]} in viewers."""
    pool = []
    for comp in case.get("comparables") or []:
        feat = comparable_features(comp)
        if feat:
            pool.append(feat)
    if len(pool) < MIN_POOL:
        raise ValueError("too few usable comparables")

    history = case.get("history_hourly") or case.get("history") or []
    peak_case = prev_day_peak(case.get("day_start"), history)
    trough_case = (p1 / peak_case) if peak_case else None

    # Clamp the case's features to the pool's observed range: the regression is an
    # interpolation device, and a launch bigger than every comparable must not be
    # extrapolated into an ever-flatter (or steeper) curve.
    log_h1s = sorted(math.log(f["h1"]) for f in pool)
    log_p1 = min(max(math.log(p1), log_h1s[0]), log_h1s[-1])
    log_trough = None
    pool_troughs = sorted(math.log(f["trough"]) for f in pool if f["trough"])
    if trough_case and pool_troughs:
        log_trough = min(max(math.log(trough_case), pool_troughs[0]), pool_troughs[-1])

    rows = {}
    for hour in hours:
        offsets = hour_forecast(pool, log_p1, log_trough, hour, zs)
        if offsets is None:
            raise ValueError("an hour has too few comparables")
        rows[hour] = [p1 * math.exp(off) for off in offsets]
    return rows


def persistence_forecast(p1, hours, zs):
    """The fallback: every hour at hour 1's print, spread widening like sqrt(hour - 1)."""
    return {
        h: [p1 * math.exp(z * 0.15 * math.sqrt(max(0, h - 1))) for z in zs]
        for h in hours
    }


def first_hour_print(case):
    """Hour 1's print from the case's known rows; 1.0 when there is none."""
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    for row in known:
        try:
            if row.get("hour") == 1 and row.get("viewers") is not None:
                return max(1.0, float(row["viewers"]))
        except Exception:
            continue
    return 1.0


def forecast_hours(case):
    """The hours to forecast, from the case's forecast rows (default 2..24)."""
    hours = []
    for row in case.get("forecast") or []:
        try:
            hours.append(int(row["hour"]))
        except Exception:
            continue
    return sorted(set(hours)) if hours else list(range(2, 25))


def round2(value):
    """Round to 2 decimals, half up; whole numbers come back as int (JSON writes 1240)."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 100
    return int(result) if float(result).is_integer() else result


def clean_row(values):
    """Make one row safe to write: finite, in (0, 1e9], non-decreasing, rounded."""
    cleaned = []
    for value in values:
        if not math.isfinite(value):
            value = 1.0
        value = min(max(float(value), 0.01), 1_000_000_000.0)
        if cleaned and value < cleaned[-1]:
            value = cleaned[-1]
        cleaned.append(value)
    return [round2(v) for v in cleaned]


def forecast_case(case):
    """Forecast one case, falling back to persistence on any model failure."""
    levels = case.get("levels") or LEVELS_DEFAULT
    dist = NormalDist()
    zs = [dist.inv_cdf(float(t)) for t in levels]
    p1 = first_hour_print(case)
    hours = forecast_hours(case)
    try:
        rows = model_forecast(case, p1, hours, zs)
    except Exception:
        rows = persistence_forecast(p1, hours, zs)
    return {"hours": [{"hour": h, "q": clean_row(rows[h])} for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A case we cannot even parse still gets a valid, if flat, forecast file
            # whenever its stem is known; nothing here may take the whole run down.
            try:
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
            except Exception:
                continue
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
