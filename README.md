# Sunday Edge

NFL betting model that compares FanDuel's lines to the rest of the market and to projections built on
advanced stats, then publishes the plays with the most supporting evidence to a dashboard.

## What runs, and when

GitHub Actions (`.github/workflows/weekly.yml`), all times 6:00 AM Central:

| Day      | Odds pulled                                   | Why                                          |
|----------|-----------------------------------------------|----------------------------------------------|
| Tuesday  | none                                          | Grade last week's plays, refresh projections |
| Thursday | game lines + props for Thursday night         | Thursday game plays, early look at the week  |
| Sunday   | game lines + props for Sunday/Monday games    | Final plays with the week's injury news      |

About 110 of The Odds API's 500 free monthly credits per week. Props are skipped automatically if the
month's remaining credits can't cover them.

A Claude scheduled task republishes `site/sunday_edge.html` to the dashboard after each run.

Manual run: **Actions → Sunday Edge weekly run → Run workflow** (choose `none`, `games`, or `full` odds).

## Setup

Repository secret `ODDS_API_KEY` (Settings → Secrets and variables → Actions) with a key from the-odds-api.com.

## Data

All from nflverse (github.com/nflverse/nflverse-data), which mirrors:

- NFL play-by-play with EPA, win probability, CPOE, expected pass rate (the data NFL Savant publishes)
- Next Gen Stats: separation, cushion, YAC over expected, rush yards over expected, 8+ box rate, time to throw, CPOE, aggressiveness
- Pro Football Reference advanced stats: pressures, bad throws, drops, broken tackles, yards before/after contact
- Snap counts, depth charts, injury reports, FTN charting

## How picks are chosen

1. **Fair price.** Every other book's price is de-vigged and combined (Pinnacle weighted 3x). For props, that
   consensus is blended 60/40 with the model's projection.
2. **Edge.** Expected value at FanDuel's price must be at least 2%.
3. **Support.** At least 3 independent signals must agree, one of them being the market or the model:
   - Game lines: other books, stats-model lean, key numbers (3, 7), rest, wind, and early-down EPA / explosive-play / pressure matchups.
   - Props: other books, projection, role trend (target, carry, or snap share), opponent allowance at that position,
     implied team total or spread, and Next Gen Stats / PFR efficiency.
4. **Units.** 0.5 to 2u by edge, +0.5u when 5+ signals agree, +0.5u when 5+ books priced it; capped at 1u for
   long shots, 0.5u for Questionable players, 3u for any play.

Every play is logged in `data/pick_log.csv` at FanDuel's price and graded after the game.

## Models (`pipeline/`)

- `features.py`, `model.py`: team ratings (EPA, success rate, early-down EPA, explosive rate, PFR pressure,
  CPOE, pass rate over expected), margin/total models, walk-forward backtest vs closing lines.
- `props2.py`: gradient-boosted usage x efficiency models per stat with calibrated outcome ranges; `props.py`
  is the simple weighted-average baseline it is blended with.
- `picks.py`: consensus pricing, signals, selection, unit sizing. `track.py`: logging and grading.
- `run.py`: the weekly pipeline. `build_dashboard.py` + `dashboard_template.html`: the page.

Backtests (2022-2025, each season predicted only from earlier seasons): the game model alone does not beat
closing spreads or moneylines, so game plays require FanDuel to be off the market. Prop projections beat a
simple weighted average by 1-6% on every stat, and their probabilities are calibrated.
