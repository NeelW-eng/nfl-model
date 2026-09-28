"""Log every published pick and grade it once the game is played."""
import os, json
import pandas as pd, numpy as np
from paths import DATA, CUR
import model

LOG = os.path.join(DATA, 'pick_log.csv')
COLS = ['season', 'week', 'logged', 'kickoff', 'market', 'pick', 'game', 'team', 'home', 'away', 'point', 'side', 'stat',
        'player_id', 'line', 'price', 'units', 'p', 'ev', 'support', 'confidence', 'tier', 'result', 'profit_units']


def log_picks(season, week, picks, pulled_at):
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=COLS)
    log = log.reindex(columns=COLS)
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


def locked_count(season, week):
    """Picks this week that already kicked off (they use up slots of the weekly 20)."""
    if not os.path.exists(LOG):
        return 0, []
    log = pd.read_csv(LOG).reindex(columns=COLS)
    kick = pd.to_datetime(log.kickoff, utc=True, errors='coerce')
    d = log[(log.season == season) & (log.week == week) & (kick <= pd.Timestamp.now(tz='UTC'))]
    return len(d), d[['pick', 'price', 'units', 'confidence', 'tier', 'result']].to_dict('records')


TIER_ORDER = ['Very likely', 'Likely', 'Favored', 'Slight edge', 'Coin flip']


def record(log):
    if log is None or log.empty:
        return None
    log = log.reindex(columns=COLS).copy()
    log['kick'] = pd.to_datetime(log.kickoff, utc=True, errors='coerce')
    log = log.sort_values(['kick', 'logged']).reset_index(drop=True)
    d = log[log.result.isin(['win', 'loss', 'push'])].copy()
    out = {'pending': int(log.result.isna().sum()), 'bets': int(len(d))}
    if len(d):
        d['cum'] = d.profit_units.astype(float).cumsum().round(2)
        out.update(wins=int((d.result == 'win').sum()), losses=int((d.result == 'loss').sum()),
                   pushes=int((d.result == 'push').sum()), units_risked=float(d.units.sum()),
                   profit_units=round(float(d.profit_units.sum()), 2))
        out['roi'] = round(out['profit_units'] / out['units_risked'], 4) if out['units_risked'] else 0
        dec = d[d.result != 'push']
        out['win_rate'] = round(float((dec.result == 'win').mean()), 4) if len(dec) else None
        out['expected_win_rate'] = round(float(dec.p.mean()), 4) if len(dec) else None
        out['series'] = [dict(n=i + 1, week=int(r.week), pick=r.pick, result=r.result, units=float(r.units),
                              profit=round(float(r.profit_units), 2), cum=float(r.cum))
                         for i, r in enumerate(d.itertuples())]
        tiers = []
        for t_ in TIER_ORDER:
            x = dec[dec.tier == t_]
            if len(x):
                tiers.append(dict(tier=t_, bets=int(len(x)), wins=int((x.result == 'win').sum()),
                                  actual=round(float((x.result == 'win').mean()), 4), expected=round(float(x.p.mean()), 4),
                                  profit_units=round(float(d[d.tier == t_].profit_units.sum()), 2)))
        out['tiers'] = tiers
        # week by week: wins, losses, and the wins the model's percentages predicted
        wk = []
        for (s_, w_), x in d.groupby(['season', 'week']):
            xd = x[x.result != 'push']
            wk.append(dict(season=int(s_), week=int(w_), wins=int((x.result == 'win').sum()),
                           losses=int((x.result == 'loss').sum()), pushes=int((x.result == 'push').sum()),
                           expected=round(float(xd.p.sum()), 2), profit_units=round(float(x.profit_units.sum()), 2)))
        out['weeks'] = wk
        # verdict: are actual wins in line with predicted wins? (z-score of the difference)
        exp_w = float(dec.p.sum()); var = float((dec.p * (1 - dec.p)).sum())
        out['expected_wins'] = round(exp_w, 1)
        out['z'] = round((out['wins'] - exp_w) / var ** 0.5, 2) if var > 0 else 0.0
        out['by_market'] = {m: {'bets': int(len(x)), 'profit_units': round(float(x.profit_units.sum()), 2),
                                'wins': int((x.result == 'win').sum()), 'losses': int((x.result == 'loss').sum())}
                            for m, x in d.groupby('market')}
    out['all'] = [dict(season=int(r.season), week=int(r.week), pick=r.pick, market=r.market, price=int(r.price),
                       units=float(r.units), confidence=None if pd.isna(r.confidence) else float(r.confidence),
                       tier=None if pd.isna(r.tier) else r.tier, p=float(r.p),
                       result=None if pd.isna(r.result) else r.result,
                       profit=None if pd.isna(r.profit_units) else round(float(r.profit_units), 2))
                  for r in log.itertuples()][::-1]
    return out
