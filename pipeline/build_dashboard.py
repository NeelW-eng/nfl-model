"""Fill the dashboard template with this week's picks + backtest summary."""
import json, sys, os
import pandas as pd
from paths import DATA, CUR, FIRST, HERE, SITE
d = json.load(open(f'{DATA}/week_picks.json'))
d['backtest'] = json.load(open(f'{DATA}/backtest_summary.json'))
g = pd.read_csv(f'{DATA}/games.csv')
done = g[(g.season == CUR) & g.home_score.notna()]
last = done.sort_values('gameday').iloc[-1]
d['stats_through'] = f"Week {int(last.week)}, {last.gameday}"
import math


def clean(x):
    """Strict JSON: NaN/Infinity become null (the page database and browsers reject them)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, 'item'):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


d = clean(d)
json.dump(d, open(f'{DATA}/board.json', 'w'), allow_nan=False)  # the page's live database copy
html = open(os.path.join(HERE, 'dashboard_template.html')).read().replace('/*DATA*/null', json.dumps(d, allow_nan=False))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SITE, 'sunday_edge.html')
open(out, 'w').write(html)
print('wrote', out, len(html))
