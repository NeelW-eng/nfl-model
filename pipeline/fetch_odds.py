"""Pull NFL lines from FanDuel and every other book on The Odds API (US + Pinnacle), for game lines
and player props. Runs in GitHub Actions with the ODDS_API_KEY repository secret.

Credit budget (free tier = 500/month):
  game lines (all books, US + Pinnacle)       6 credits
  props                                       6 credits per game pulled
  --props-hours N   only pull props for games kicking off in the next N hours (default 30)
  --no-props        game lines only; keeps the last props pull
Props are skipped automatically if the month's remaining credits can't cover them (keeps a 20-credit reserve).
"""
import json, os, sys, urllib.request, urllib.parse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.environ.get('NFL_DATA', os.path.join(ROOT, 'data')), 'fanduel_odds.json')
BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
PROP_MARKETS = ['player_pass_yds', 'player_pass_tds', 'player_rush_yds',
                'player_reception_yds', 'player_receptions', 'player_anytime_td']


def get(path, **params):
    params.update(apiKey=os.environ['ODDS_API_KEY'], oddsFormat='american')
    url = f'{BASE}{path}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r), r.headers.get('x-requests-remaining')


def main():
    if not os.environ.get('ODDS_API_KEY'):
        print('ODDS_API_KEY secret not set - skipping odds pull'); return
    props = '--no-props' not in sys.argv
    hours = float(sys.argv[sys.argv.index('--props-hours') + 1]) if '--props-hours' in sys.argv else 30
    now = datetime.datetime.now(datetime.timezone.utc)
    games, left = get('/odds', regions='us,eu', markets='h2h,spreads,totals')
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out = {'pulled_at': now.isoformat(), 'games': games,
           'props': prev.get('props', {}), 'props_pulled_at': prev.get('props_pulled_at')}
    if props:
        cut = now + datetime.timedelta(hours=hours)
        todo = [g for g in games
                if now < datetime.datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) < cut]
        need = 6 * len(todo)
        if left is not None and int(float(left)) - need < 20:
            print(f'Only {left} credits left this month; skipping props ({need} needed).')
            todo = []
        out['props'] = {}
        for g in todo:
            try:
                d, left = get(f"/events/{g['id']}/odds", regions='us', markets=','.join(PROP_MARKETS))
                d.setdefault('commence_time', g['commence_time'])
                out['props'][g['id']] = d
            except Exception as e:
                print('props failed', g['away_team'], '@', g['home_team'], e)
        out['props_pulled_at'] = now.isoformat()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'))
    print(f"{len(games)} games, props for {len(out['props'])} games. Credits left: {left}")


if __name__ == '__main__':
    main()
