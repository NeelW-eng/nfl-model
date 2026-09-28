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
html = open(os.path.join(HERE, 'dashboard_template.html')).read().replace('/*DATA*/null', json.dumps(d, default=float))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SITE, 'sunday_edge.html')
open(out, 'w').write(html)
print('wrote', out, len(html))
