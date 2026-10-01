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
    """ESPN's league-wide injury feed (credible, updated through the week). Failure is non-fatal."""
    try:
        req = urllib.request.Request('https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries',
                                     headers={'User-Agent': 'Mozilla/5.0 (sunday-edge)'})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = r.read()
        import json
        n = sum(len(t.get('injuries', [])) for t in json.loads(d).get('injuries', []))
        open(os.path.join(DATA, 'injuries_espn.json'), 'wb').write(d)
        return f'ok injuries_espn.json ({n} listings)'
    except Exception as e:
        return f'ESPN injury feed unavailable: {e}'


if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True)
    force = '--force' in sys.argv
    with cf.ThreadPoolExecutor(6) as ex:
        for dst, st in ex.map(lambda j: get(j, force), jobs()):
            print(st, dst)
    print(espn_injuries())
