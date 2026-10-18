"""Pull Underdog pick'em lines plus 9 major sportsbooks (for a de-vigged consensus) from The Odds API.

Runs in GitHub Actions with the ODDS_API_KEY secret. Writes data/odds.json.

Cost: The Odds API bills 1 credit per market per 10 bookmakers, so one request covering Underdog and
9 sportsbooks costs 1 credit per market:
  game lines (moneyline, spread, total)              3 credits
  player props, per game (4 markets)                 4 credits
Options:
  --props-hours N   props for games starting in the next N hours
  --props-week      props for every game in the current NFL week
  --no-props        game lines only (keeps earlier prop pulls)
  --force           ignore the "props pulled in the last 18 hours" guard
  --backup          (late backup runs) do nothing if odds were pulled in the last 12 hours
A reserve (ODDS_RESERVE, default 10) is always kept; games are pulled soonest-first until the budget runs out.
"""
import json, os, sys, urllib.request, urllib.parse, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('NFL_DATA', os.path.join(ROOT, 'data'))
OUT = os.path.join(DATA, 'odds.json')
BASE = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl'
BOOKS = 'underdog,draftkings,fanduel,betmgm,williamhill_us,espnbet,fanatics,betrivers,bovada,betonlineag'
GAME_BOOKS = 'draftkings,fanduel,betmgm,williamhill_us,espnbet,fanatics,betrivers,pinnacle,bovada,betonlineag'
# pass TDs dropped: Underdog posts few of them and each market costs 1 credit per game
PROP_MARKETS = ['player_pass_yds', 'player_rush_yds', 'player_reception_yds', 'player_receptions']
RESERVE = int(os.environ.get('ODDS_RESERVE', 10))
FRESH_HOURS = 18     # a game's props pulled within this window are not pulled again (unless --force)
BACKUP_HOURS = 12    # --backup: skip the whole pull if odds were pulled this recently


def get(path, **params):
    params.update(apiKey=os.environ['ODDS_API_KEY'], oddsFormat='american')
    url = f'{BASE}{path}?{urllib.parse.urlencode(params)}'
    with urllib.request.urlopen(url, timeout=30) as r:
        left = r.headers.get('x-requests-remaining')
        return json.load(r), (int(float(left)) if left is not None else None)


def live_credits():
    """Remaining credits right now, from the free /sports endpoint (costs nothing), so a new or upgraded
    key is seen immediately instead of trusting the last saved count."""
    try:
        url = 'https://api.the-odds-api.com/v4/sports?' + urllib.parse.urlencode({'apiKey': os.environ['ODDS_API_KEY']})
        with urllib.request.urlopen(url, timeout=30) as r:
            v = r.headers.get('x-requests-remaining')
            return int(float(v)) if v is not None else None
    except Exception as e:
        print('could not read live credit balance:', e)
        return None


def ts(s):
    return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))


def nfl_week_end(now):
    """Next Tuesday 08:00 UTC: the end of the current NFL week (Thursday through Monday night)."""
    d = (1 - now.weekday()) % 7 or 7
    return (now + datetime.timedelta(days=d)).replace(hour=8, minute=0, second=0, microsecond=0)


def main():
    if not os.environ.get('ODDS_API_KEY'):
        print('ODDS_API_KEY secret not set - skipping odds pull'); return
    a = sys.argv
    now = datetime.datetime.now(datetime.timezone.utc)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if '--backup' in a and prev.get('pulled_at') and ts(prev['pulled_at']) > now - datetime.timedelta(hours=BACKUP_HOURS):
        print('Backup run: odds already pulled at', prev['pulled_at'], '- skipping (no credits used).'); return
    left0 = live_credits()
    if left0 is not None and left0 < RESERVE + 3:
        print(f'Only {left0} credits left this month - skipping the odds pull.'); return
    games, left = get('/odds', bookmakers=GAME_BOOKS, markets='h2h,spreads,totals')
    out = {'pulled_at': now.isoformat(), 'games': games, 'props': prev.get('props', {}),
           'props_pulled_at': prev.get('props_pulled_at'), 'credits_left': left}
    # drop props for games that finished more than a day ago
    out['props'] = {k: v for k, v in out['props'].items()
                    if ts(v.get('commence_time', '2000-01-01T00:00:00Z')) > now - datetime.timedelta(days=1)}
    if '--no-props' not in a:
        if '--props-week' in a:
            cut = nfl_week_end(now)
        else:
            hours = float(a[a.index('--props-hours') + 1]) if '--props-hours' in a else 30
            cut = now + datetime.timedelta(hours=hours)
        todo = sorted((g for g in games if now < ts(g['commence_time']) < cut), key=lambda g: g['commence_time'])
        if '--force' not in a:
            todo = [g for g in todo if not (g['id'] in out['props'] and out['props'][g['id']].get('pulled_at') and
                    ts(out['props'][g['id']]['pulled_at']) > now - datetime.timedelta(hours=FRESH_HOURS))]
        cost = len(PROP_MARKETS)
        pulled = 0
        for g in todo:
            if left is not None and left - cost < RESERVE:
                print(f'Stopping: {left} credits left this month (keeping {RESERVE} in reserve).'); break
            try:
                d, left = get(f"/events/{g['id']}/odds", bookmakers=BOOKS, markets=','.join(PROP_MARKETS))
                d['commence_time'] = g['commence_time']
                d['pulled_at'] = now.isoformat()
                out['props'][g['id']] = d
                pulled += 1
            except Exception as e:
                print('props failed', g['away_team'], '@', g['home_team'], e)
        if pulled:
            out['props_pulled_at'] = now.isoformat()
        print(f'props pulled for {pulled} of {len(todo)} games')
    out['credits_left'] = left
    os.makedirs(DATA, exist_ok=True)
    json.dump(out, open(OUT, 'w'))
    print(f"{len(games)} games with lines, props on file for {len(out['props'])} games. Credits left: {left}")


if __name__ == '__main__':
    main()
