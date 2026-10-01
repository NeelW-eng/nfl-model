"""Download everything the model uses from nflverse (mirrors of PFR, Next Gen Stats, FTN, NFL play-by-play)."""
import os, sys, urllib.request, concurrent.futures as cf
from paths import DATA, CUR, FIRST

B = 'https://github.com/nflverse/nflverse-data/releases/download'


def jobs():
    j = [('schedules/games.csv', 'games.csv'), ('players/players.parquet', 'players.parquet')]
    j += [(f'nextgen_stats/ngs_{k}.parquet', f'ngs_{k}.parquet') for k in ('passing', 'receiving', 'rushing')]
    for y in range(FIRST, CUR + 1):
        j += [(f'pbp/play_by_play_{y}.parquet', f'pbp_{y}.parquet'),
              (f'stats_player/stats_player_week_{y}.parquet', f'ps_{y}.parquet'),
              (f'snap_counts/snap_counts_{y}.parquet', f'snaps_{y}.parquet')]
        j += [(f'pfr_advstats/advstats_week_{k}_{y}.parquet', f'pfr_{k}_{y}.parquet') for k in ('pass', 'rec', 'rush', 'def')]
    j.append((f'injuries/injuries_{CUR}.parquet', f'inj_{CUR}.parquet'))
    j.append((f'weekly_rosters/roster_weekly_{CUR}.parquet', f'roster_weekly_{CUR}.parquet'))
    return j


def get(job, force):
    src, dst = job
    path = os.path.join(DATA, dst)
    # past seasons never change: only re-download current-season files and the rolling tables
    stable = any(str(y) in dst for y in range(FIRST, CUR)) and os.path.exists(path)
    if stable and not force:
        return dst, 'cached'
    urllib.request.urlretrieve(f'{B}/{src}', path + '.tmp')
    os.replace(path + '.tmp', path)
    return dst, 'ok'


def espn_injuries():
    """ESPN's league-wide injury feed (credible, updated through the week). Failure is non-fatal; the
    outcome is written to data/sources_status.json so it shows up in the repo."""
    import json, datetime
    tried = []
    for url in ('https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries',
                'https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/injuries'):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)',
                                                       'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = r.read()
            full = json.loads(d)
            slim = {'timestamp': full.get('timestamp'), 'injuries': []}
            for t in full.get('injuries', []):
                rows = []
                for e in t.get('injuries', []):
                    a = e.get('athlete') or {}
                    href = ((a.get('links') or [{}])[0] or {}).get('href', '')
                    aid = a.get('id') or (href.split('/id/')[1].split('/')[0] if '/id/' in href else None)
                    rows.append({'athlete': {'id': aid, 'displayName': a.get('displayName')},
                                 'status': e.get('status'), 'shortComment': (e.get('shortComment') or '')[:240],
                                 'date': e.get('date')})
                slim['injuries'].append({'displayName': t.get('displayName'), 'injuries': rows})
            n = sum(len(t['injuries']) for t in slim['injuries'])
            json.dump(slim, open(os.path.join(DATA, 'injuries_espn.json'), 'w'))
            tried.append(dict(url=url, ok=True, listings=n))
            break
        except Exception as e:
            tried.append(dict(url=url, ok=False, error=str(e)[:200]))
    json.dump({'checked': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'espn': tried},
              open(os.path.join(DATA, 'sources_status.json'), 'w'), indent=1)
    return tried


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    force = '--force' in sys.argv
    with cf.ThreadPoolExecutor(6) as ex:
        for dst, st in ex.map(lambda j: get(j, force), jobs()):
            print(st, dst)
    print(espn_injuries())
