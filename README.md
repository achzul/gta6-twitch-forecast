![Live rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/achzul/gta6-twitch-forecast/rank.svg)
![Live reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/achzul/gta6-twitch-forecast/reward.svg)
![make.fun bounty badge](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)

Entry for the make.fun bounty: [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch CCU forecast

`python3 main.py` (Python 3.13, standard library only, no network) reads each case file in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one forecast per case to `$MAKEFUN_OUTPUT_DIR`
(else `/output`): for hours 2–24 of the launch day, five quantiles (5%, 25%, 50%, 75%, 95%) of
the game's Twitch category viewers.

## The model

A launch day is a wave that rolls at midnight local time — New Zealand first — so by the end of
hour 1 (01:00 UTC) the day's shape is already constrained: hour 1 catches the Asia/Pacific and
European midnight wave, the curve dips as Europe sleeps, then builds again into the American
evening. The program learns that shape from the case's own comparable launches:

1. Each comparable is reduced to the ratio of every release-day hour to its own hour 1.
2. For each target hour, a weighted regression of the log ratio is fit across the pool on two
   launch features: the log of the launch's hour-1 size, and the log *trough depth* — hour 1
   divided by the previous day's peak, which measures how much of the wave had already arrived
   before the day began.
3. The case's median is its hour-1 print times the regression prediction at the case's features,
   with the features clamped to the pool's observed range so an out-of-scale launch is never
   extrapolated. The spread is the regression's residual spread, floored at 0.20 log units and
   scaled by 0.70.
4. Comparables are weighted by kind: early-access-contaminated ×0.25, weak ×0.5, and launches
   that did not roll at midnight local ×0.10. The target game's own pre-release history (tier
   `target-history`) is never used as a launch.

Any failure in the model path falls back per case to a persistence forecast (every hour at hour
1's print, spread widening like √hour), so the program always writes a valid file and never
crashes.

## Practice results

Scored locally with the contest's mean pinball loss on the log scale, against the historical
counts for the four practice launches (see BUILD_NOTES.md):

| forecast | p00 Wolverine | p01 RDR2 | p02 God of War Ragnarök | p03 Spider-Man 2 | mean |
|---|---|---|---|---|---|
| this model | 0.1399 | 0.1191 | 0.1183 | 0.2211 | **0.1496** |
| persistence baseline | 0.7436 | 0.2868 | 0.2146 | 0.3033 | 0.3871 |
| flat (all quantiles = hour 1) | 1.0605 | 0.5034 | 0.3509 | 0.4969 | 0.6029 |

The model's 5%–95% band covered 93.5% of the 92 practice hours.
