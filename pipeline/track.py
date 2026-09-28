"""Log every published pick and grade it once the game is played."""
import os, json
import pandas as pd, numpy as np
from paths import DATA, CUR
import model

LOG = os.path.join(DATA, 'pick_log.csv')
COLS = ['season', 'week', 'logged', 'kickoff', 'market', 'pick', 'game', 'team', 'home', 'away', 'point', 'side', 'stat',
        'player_id', 'line', 'price', 'units', 'p', 'ev', 'support', 'result', 'profit_units']


def log_picks(season, week, picks, pulled_at):
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    # a newer pull replaces this week's picks that haven't kicked off yet; started games keep their logged picks
    now = pd.Timestamp.now(tz='UTC')
    kick = pd.to_datetime(log.kickoff, utc=True, errors='coerce')
    keep = ~((log.season == season) & (log.week == week) & log.result.isna() & (kick.isna() | (kick > now)))
    rows = []
    for p in picks:
        side = p.get('side') or ('Over' if p['pick'].startswith('Over') else 'Under' if p['pick'].startswith('Under') else None)
        rows.append({**{c: p.get(c) for c in COLS}, 'season': season, 'week': week, 'logged': pulled_at, 'side': side})
    new = pd.DataFrame(rows, columns=COLS)
    out = pd.concat([log[keep], new], ignore_index=True) if len(new) else log[keep]
    out.to_csv(LOG, index=False)


def grade():
    if not os.path.exists(LOG):
        return pd.DataFrame(columns=COLS)
    log = pd.read_csv(LOG)
    g = pd.read_csv(f'{DATA}/games.csv')
    g = g[g.home_score.notna()]
    ps = pd.read_parquet(f'{DATA}/ps_{CUR}.parquet')
    ps['anytime_td'] = (ps.rushing_tds.fillna(0) + ps.receiving_tds.fillna(0)) > 0
    for i, r in log[log.result.isna()].iterrows():
        res = None
        if r.market in ('Spread', 'Total', 'Moneyline'):
            gm = g[(g.season == r.season) & (g.week == r.week) & (g.home_team == r.home) & (g.away_team == r.away)]
            if gm.empty:
                continue
            gm = gm.iloc[0]
            margin = gm.home_score - gm.away_score
            tot = gm.home_score + gm.away_score
            if r.market == 'Spread':
                m = margin if r.team == r.home else -margin
                v = m + r.point
                res = 'push' if v == 0 else 'win' if v > 0 else 'loss'
            elif r.market == 'Total':
                v = (tot - r.point) * (1 if r.side == 'Over' else -1)
                res = 'push' if v == 0 else 'win' if v > 0 else 'loss'
            else:
                m = margin if r.team == r.home else -margin
                res = 'push' if m == 0 else 'win' if m > 0 else 'loss'
        else:
            st = ps[(ps.season == r.season) & (ps.week == r.week) & (ps.player_id == r.player_id)]
            played_week = g[(g.season == r.season) & (g.week == r.week)]
            if st.empty:
                # player didn't record a stat line: void once his team's game is final
                if len(played_week) and (played_week.home_team.isin([r.team]) | played_week.away_team.isin([r.team])).any():
                    res = 'void'
                else:
                    continue
            else:
                val = st.iloc[0][r.stat] if r.stat != 'anytime_td' else float(st.iloc[0].anytime_td)
                if r.stat == 'anytime_td':
                    res = 'win' if val else 'loss'
                else:
                    v = (val - r.line) * (1 if r.side == 'Over' else -1)
                    res = 'push' if v == 0 else 'win' if v > 0 else 'loss'
        log.at[i, 'result'] = res
        log.at[i, 'profit_units'] = (r.units * float(model.payout(r.price)) if res == 'win'
                                     else -r.units if res == 'loss' else 0.0)
    log.to_csv(LOG, index=False)
    return log


def record(log):
    d = log[log.result.isin(['win', 'loss', 'push'])]
    if d.empty:
        return None
    out = {'bets': int(len(d)), 'wins': int((d.result == 'win').sum()), 'losses': int((d.result == 'loss').sum()),
           'pushes': int((d.result == 'push').sum()), 'units_risked': float(d.units.sum()),
           'profit_units': round(float(d.profit_units.sum()), 2)}
    out['roi'] = round(out['profit_units'] / out['units_risked'], 4) if out['units_risked'] else 0
    out['by_market'] = {m: {'bets': int(len(x)), 'profit_units': round(float(x.profit_units.sum()), 2)}
                        for m, x in d.groupby('market')}
    out['recent'] = d.sort_values(['season', 'week']).tail(25)[['week', 'pick', 'price', 'units', 'result', 'profit_units']].to_dict('records')
    return out
