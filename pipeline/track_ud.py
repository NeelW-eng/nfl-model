"""Season tracker for Underdog picks: logs the Top 10 and each game's top 4, locks them at kickoff,
grades them from box scores (a player who doesn't play is void, as on Underdog)."""
import os, datetime
import numpy as np, pandas as pd
from paths import DATA, CUR

LOG = os.path.join(DATA, 'picks_underdog.csv')
COLS = ['season', 'week', 'logged', 'kickoff', 'game_id', 'game', 'player', 'player_id', 'team', 'opp', 'pos',
        'stat', 'stat_label', 'side', 'word', 'line', 'pick', 'p', 'n_books', 'support', 'n_signals',
        'top10', 'rank', 'game4', 'result', 'actual', 'market', 'home', 'away', 'fair_odds', 'gl', 'hp']


def _load():
    if not os.path.exists(LOG):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(LOG).reindex(columns=COLS)
    d['result'] = d['result'].astype(object)
    d['actual'] = d['actual'].astype(object)   # numbers for player picks, final score text for game lines
    d['top10'] = d.top10.fillna(False).astype(bool)
    d['game4'] = d.game4.fillna(False).astype(bool)
    d['gl'] = d.gl.fillna(False).astype(bool)
    d['hp'] = d.hp.fillna(False).astype(bool)
    d['market'] = d.market.fillna('Prop')
    return d


def _key(r):
    if r.get('market') in ('Spread', 'Total', 'Moneyline'):
        ln = r.get('line')
        return (r['game_id'], r['market'], r.get('team') if r['market'] != 'Total' else r['side'],
                None if ln is None or (isinstance(ln, float) and np.isnan(ln)) else float(ln))
    return (r['player_id'], r['stat'], r['side'], float(r['line']))


def locked(season, week):
    """Picks of this week whose game has kicked off: they stay on the board as logged."""
    d = _load()
    if d.empty:
        return [], [], [], []
    kick = pd.to_datetime(d.kickoff, utc=True, errors='coerce')
    d = d[(d.season == season) & (d.week == week) & (kick <= pd.Timestamp.now(tz='UTC'))]
    rows = [dict(r, started=True, signals=[], injury=None) for r in d.to_dict('records')]
    for r in rows:
        r['line'] = None if pd.isna(r['line']) else float(r['line'])
        r['history'] = 99
    return ([r for r in rows if r['top10']], [r for r in rows if r['game4']], [r for r in rows if r['gl']],
            [r for r in rows if r['hp']])


def log_picks(season, week, top, games, pulled_at, gl=(), hp=()):
    d = _load()
    kick = pd.to_datetime(d.kickoff, utc=True, errors='coerce')
    now = pd.Timestamp.now(tz='UTC')
    keep = ~((d.season == season) & (d.week == week) & d.result.isna() & (kick > now))
    d = d[keep]
    have = {_key(r) for r in d[(d.season == season) & (d.week == week)].to_dict('records')}
    rows = {}
    for c in top:
        if c.get('started'):
            continue
        rows[_key(c)] = dict({k: c.get(k) for k in COLS}, top10=True, rank=c['rank'], game4=False, gl=False, market='Prop')
    for g in games:
        for c in g['picks']:
            if c.get('started'):
                continue
            k = _key(c)
            if k in rows:
                rows[k]['game4'] = True
            else:
                rows[k] = dict({k2: c.get(k2) for k2 in COLS}, top10=False, rank=None, game4=True, gl=False, market='Prop')
    for c in gl:
        if c.get('started'):
            continue
        rows[_key(c)] = dict({k2: c.get(k2) for k2 in COLS}, top10=False, rank=None, game4=False, gl=True,
                             stat=c['market'], stat_label=c['market'])
    for c in hp:
        if c.get('started'):
            continue
        k = _key(c)
        if k in rows:
            rows[k]['hp'] = True
        else:
            is_gl = c.get('market') in ('Spread', 'Total', 'Moneyline')
            rows[k] = dict({k2: c.get(k2) for k2 in COLS}, top10=False, rank=None, game4=False, gl=False, hp=True,
                           market=c.get('market') or 'Prop',
                           **({'stat': c['market'], 'stat_label': c['market']} if is_gl else {}))
    for r in rows.values():
        r.setdefault('hp', False)
        r['hp'] = bool(r.get('hp'))
    new = [dict(r, season=season, week=week, logged=pulled_at, result=None, actual=None)
           for k, r in rows.items() if k not in have]
    if new:
        d = pd.concat([d, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    d.to_csv(LOG, index=False)


def grade():
    d = _load()
    if d.empty:
        return d
    g = pd.read_csv(f'{DATA}/games.csv')
    done = g[g.home_score.notna()]
    finals = {(r.season, r.week, t) for r in done.itertuples() for t in (r.home_team, r.away_team)}
    stats = {}
    score = {(int(x.season), int(x.week), x.home_team, x.away_team): (x.home_score, x.away_score) for x in done.itertuples()}
    for i, r in d[d.result.isna() & d.gl].iterrows():
        sc = score.get((int(r.season), int(r.week), r.home, r.away))
        if sc is None:
            continue
        margin = sc[0] - sc[1]
        if r.market == 'Spread':
            v = (margin if r.team == r.home else -margin) + float(r.line)
        elif r.market == 'Total':
            v = (sc[0] + sc[1] - float(r.line)) * (1 if r.side == 'Over' else -1)
        else:
            v = margin if r.team == r.home else -margin
        d.at[i, 'result'] = 'push' if v == 0 else 'win' if v > 0 else 'loss'
        d.at[i, 'actual'] = f"{r.away} {int(sc[1])}-{int(sc[0])} {r.home}"
    for i, r in d[d.result.isna() & ~d.gl].iterrows():
        s = int(r.season)
        if (s, int(r.week), r.team) not in finals:
            continue
        if s not in stats:
            p = f'{DATA}/ps_{s}.parquet'
            stats[s] = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
        ps = stats[s]
        row = ps[(ps.week == r.week) & (ps.player_id == r.player_id)] if len(ps) else ps
        if row.empty:
            d.at[i, 'result'] = 'void'
            continue
        val = float(row.iloc[0][r.stat]) if pd.notna(row.iloc[0][r.stat]) else 0.0  # noqa
        v = (val - float(r.line)) * (1 if r.side == 'Over' else -1)
        d.at[i, 'result'] = 'push' if v == 0 else 'win' if v > 0 else 'loss'
        d.at[i, 'actual'] = val
    d.to_csv(LOG, index=False)
    return d


def _wl(x):
    w, l = int((x.result == 'win').sum()), int((x.result == 'loss').sum())
    return dict(wins=w, losses=l, pushes=int((x.result == 'push').sum()), voids=int((x.result == 'void').sum()),
                pending=int(x.result.isna().sum()), hit=round(w / (w + l), 4) if w + l else None,
                expected=round(float(x[x.result.isin(['win', 'loss'])].p.mean()), 4) if w + l else None)


def record(season):
    d = _load()
    d = d[d.season == season]
    out = {'breakeven': 0.54}
    for name, sub in (('top10', d[d.top10]), ('game4', d[d.game4]), ('gl', d[d.gl]), ('hp', d[d.hp])):
        weeks = [dict(week=int(w), **_wl(x)) for w, x in sub.groupby('week')]
        out[name] = dict(season=_wl(sub), weeks=weeks)
    # results keyed for marking cards
    out['results'] = {}
    for r in d.itertuples():
        if not isinstance(r.result, str):
            continue
        if r.gl:
            k = f"{r.game_id}|{r.market}|{r.team if r.market != 'Total' else r.side}"
        else:
            k = f"{r.player_id}|{r.stat}|{r.side}|{float(r.line):g}"
        out['results'][k] = dict(result=r.result, actual=None if pd.isna(r.actual) else r.actual)
    out['history'] = [dict(week=int(r.week), pick=r.pick, game=r.game, p=float(r.p), top10=bool(r.top10),
                           result=r.result if isinstance(r.result, str) else None,
                           actual=None if pd.isna(r.actual) else (r.actual if isinstance(r.actual, str) else float(r.actual)))
                      for r in d.sort_values(['week', 'kickoff', 'rank']).itertuples()][::-1]
    return out
