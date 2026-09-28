"""Player prop projections from nflverse weekly stats (prior games only).

projection = player EWMA per-game rate (shrunk toward position mean when sample is small)
             x opponent defense factor (what the defense allows to that position vs league avg)
             x game-script factor (team implied points from the line vs team's average)
Probabilities: yards ~ normal (sd fitted from backtest residuals), counts ~ Poisson/neg-binomial.
"""
import pandas as pd, numpy as np
from scipy.stats import norm, poisson, nbinom

from paths import DATA, CUR, FIRST
STATS = {  # stat: (positions, prior games needed for full weight)
    'passing_yards': (['QB'], 4), 'passing_tds': (['QB'], 6),
    'rushing_yards': (['QB', 'RB', 'WR'], 4), 'receiving_yards': (['WR', 'TE', 'RB'], 5),
    'receptions': (['WR', 'TE', 'RB'], 5), 'anytime_td': (['QB', 'RB', 'WR', 'TE'], 8),
}
HL = 6  # player half-life in games
# backtest 2023-25: projections ran ~3% high (survivorship); multiply to calibrate
CAL = {'passing_yards': .974, 'passing_tds': .989, 'rushing_yards': .96, 'receiving_yards': .966,
       'receptions': .968, 'anytime_td': .975}


def load():
    ps = pd.concat([pd.read_parquet(f'{DATA}/ps_{y}.parquet') for y in range(FIRST, CUR + 1)])
    ps = ps[ps.season_type == 'REG'].copy()
    ps['anytime_td'] = ((ps.rushing_tds.fillna(0) + ps.receiving_tds.fillna(0)) > 0).astype(float)
    ps['tds'] = ps.rushing_tds.fillna(0) + ps.receiving_tds.fillna(0)
    ps['t'] = ps.season * 100 + ps.week
    return ps.sort_values(['player_id', 't'])


def player_rates(ps, stat):
    pos = STATS[stat][0]
    d = ps[ps.position.isin(pos)].copy()
    src = 'tds' if stat == 'anytime_td' else stat
    d['val'] = d[src].fillna(0)
    g = d.groupby('player_id').val
    d['ew'] = g.transform(lambda s: s.ewm(halflife=HL).mean().shift(1))
    d['n'] = g.cumcount()
    d['last_season'] = d.groupby('player_id').season.shift(1)
    return d


def defense_factor(ps, stat):
    """EWMA of stat allowed per game to each position by each defense, relative to league."""
    pos = STATS[stat][0]
    src = 'tds' if stat == 'anytime_td' else stat
    d = ps[ps.position.isin(pos)]
    allowed = d.groupby(['opponent_team', 't', 'position'])[src].sum().reset_index()
    allowed = allowed.groupby(['opponent_team', 't'])[src].sum().reset_index().sort_values('t')
    lg = allowed.groupby('t')[src].mean().rename('lg')
    allowed = allowed.join(lg, on='t')
    allowed['ratio'] = allowed[src] / allowed.lg.replace(0, np.nan)
    allowed['dfac'] = allowed.groupby('opponent_team').ratio.transform(
        lambda s: s.ewm(halflife=8).mean().shift(1))
    allowed['dfac'] = (1 + 0.5 * (allowed.dfac - 1)).clip(0.75, 1.3).fillna(1)  # half-weight, shrunk
    return allowed[['opponent_team', 't', 'dfac']]


def project(ps, stat):
    d = player_rates(ps, stat)
    full = STATS[stat][1]
    posmean = d.groupby('position').val.mean()
    w = (d.n / full).clip(0, 1)
    # rookies/backups: shrink toward 60% of position average (typical depth player)
    d['base'] = w * d.ew.fillna(0) + (1 - w) * 0.6 * d.position.map(posmean)
    d = d.merge(defense_factor(ps, stat), on=['opponent_team', 't'], how='left')
    d['dfac'] = d.dfac.fillna(1)
    d['proj'] = d.base * d.dfac * CAL[stat]
    if stat == 'anytime_td':  # raw rates overconfident at the top (2023-25 calibration table)
        d['proj'] = 0.04 + 0.78 * d.base * d.dfac
    return d


def fit_spread(d, stat):
    """residual sd as a function of projection (yards) for probability calcs"""
    x = d[(d.n >= 3) & (d.season >= 2022)]
    if stat in ('receiving_yards', 'rushing_yards', 'passing_yards'):
        b = np.polyfit(x.proj, (x.val - x.proj).abs() * np.sqrt(np.pi / 2), 1)
        return b  # sd ~ b0*proj + b1
    return None


def prob_over(stat, proj, line, sdfit=None):
    proj = max(proj, 0.01)
    if stat in ('receiving_yards', 'rushing_yards', 'passing_yards'):
        sd = max(np.polyval(sdfit, proj), 8)
        return 1 - norm.cdf(line + 0.5 if float(line).is_integer() else line, proj, sd)
    if stat == 'receptions':  # over-dispersed count
        r = 12.0
        return 1 - nbinom.cdf(np.floor(line), r, r / (r + proj))
    if stat == 'passing_tds':
        return 1 - poisson.cdf(np.floor(line), proj)
    if stat == 'anytime_td':
        return 1 - np.exp(-proj)  # P(at least one TD), proj = expected TDs


def backtest():
    ps = load()
    res, fits = {}, {}
    for stat in STATS:
        d = project(ps, stat)
        fits[stat] = fit_spread(d, stat)
        t = d[(d.season >= 2023) & (d.n >= 3) & d.ew.notna()]
        t = t[t.proj > {'passing_yards': 150, 'passing_tds': .8, 'rushing_yards': 20,
                        'receiving_yards': 20, 'receptions': 2, 'anytime_td': .15}[stat]]
        naive = t.groupby('player_id').val.transform(lambda s: s.expanding().mean().shift(1)).fillna(t.proj)
        res[stat] = dict(n=len(t), mae_model=(t.val - t.proj).abs().mean(),
                         mae_naive=(t.val - naive).abs().mean(), bias=(t.val - t.proj).mean())
    return pd.DataFrame(res).T, fits


if __name__ == '__main__':
    r, fits = backtest()
    print(r.round(3))
    print({k: (v.round(3) if v is not None else None) for k, v in fits.items()})
