# Build notes — gta6-twitch-ccu-forecast-01

## What was built

A single-file Python program (`main.py`, standard library only) that forecasts a game launch
day's hourly Twitch viewers for hours 2–24 as five quantiles per hour, from a case file that
contains everything known at the end of hour 1: hour 1's print, the pre-launch history, and 38+
comparable past launches with their full hourly curves.

## Design process

1. **Scorer first.** Implemented the contest's scoring locally (mean pinball loss on the log
   scale, exactly as the rules define it) and verified it reproduces the house numbers: the
   persistence baseline scores 0.3871 (rules say "about 0.387") and the flat forecast 0.6029 on
   the four practice cases.
2. **Answers.** The practice answers are the practice launches' own release-day hours 2–24 from
   the same public source the kit's NOTICE cites (SullyGnome hourly averages). Each fetched
   series was verified against its case before use: its hour-1 value equals the case's `known`
   print exactly, for all four cases.
3. **Explored the data.** Every practice case shows the same skeleton: the wave starts the
   previous day (~11:00Z, New Zealand midnight), peaks in the European evening, and declines into
   hour 1; day 0 dips again midday, then peaks in the American evening (hours 17–22). Two case
   features capture where a launch sits in that skeleton: hour 1's size, and hour 1 relative to
   the previous day's peak ("trough depth": 0.35 for Wolverine, whose day then grew 9x, vs ~0.6
   for the flatter launches).
4. **Model selection.** Compared, on all four practice cases: flat and persistence references;
   pooled empirical quantiles of hour ratios (0.298 unweighted, 0.210 with similarity kernels);
   pooled log-normal fits (0.194); per-hour weighted regression of log ratios on the two features
   (0.171 at full strength). The regression won on both mean and worst case.
5. **Rejected on the evidence:** empirical quantiles (noisier tails), median recentering (worse
   mean), a day −1 same-hour anchor blend (hurt the deep-trough case), a viewers-per-channel
   regressor (gain within noise), recency weighting (hurt), asymmetric two-piece spreads (hurt).
6. **Spread calibration.** The regression's residual sd overstates the conditional spread;
   scaling it by 0.70 gives a 5–95% band covering 93.5% of practice hours — near the honest 0.90 —
   at a practice mean of 0.1496. (0.60 scored a hair better, 0.1447, but with the final being a
   single, possibly out-of-distribution launch, the slightly conservative spread was kept.)
7. **Final-regime check.** The final case's pool also contains the four practice launches. Adding
   the other three practice launches to each practice case's pool (synthesized from their case
   files plus their verified answers) improved every case: mean 0.1423.

## Final configuration

- Pool: the case's comparables, minus tier `target-history`, minus any without an hour-1 print.
- Weights: early-access-contaminated ×0.25, weak ×0.5, non-rolling unlock ×0.10.
- Per hour: weighted least squares of log(hour ratio) on [log hour-1 size, log trough depth],
  features centered on the case's, with the case's features clamped to the pool's range.
- Quantiles: hour-1 print × exp(mu + z·sd), sd = max(0.20, residual sd) × 0.70, z the standard
  normal quantiles at the case's levels.
- Fallback: any failure → persistence forecast; output rows are sanitized (finite, ≤ 1e9,
  non-decreasing) before writing.

## Validation of the shipped program

- Run through the real entry point on all four practice cases: mean pinball **0.1496**
  (p00 0.1399, p01 0.1191, p02 0.1183, p03 0.2211) vs naive flat 0.6029 and baseline 0.3871.
- Runtime: well under 0.1 s per case (limit 8 s); two consecutive runs byte-identical.
- Edge cases exercised: corrupt JSON case file, a case with no comparables, an empty input
  directory — all produce a valid forecast file or a clean no-op, never a crash.
