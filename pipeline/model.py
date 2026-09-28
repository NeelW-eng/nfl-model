"""Game models: margin (spread + moneyline) and total. Walk-forward backtest vs closing lines.

Line conventions (nflverse): spread_line > 0 means HOME favored by that many.
result = home - away. Home covers when result > spread_line.
"""
import pandas as pd, numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.stats import norm

from paths import DATA, CUR, FIRST
MARGIN_F = ['m_net', 'm_pass', 'm_rush', 'm_pts', 'm_early', 'm_expl', 'm_press', 'm_cpoe', 'rest_diff', 'neutral']
TOTAL_F = ['t_pts', 't_epa', 't_plays', 't_expl', 't_proe', 't_press', 'wind_f', 'dome']
MARGIN_SD, TOTAL_SD = 13.3, 13.0
TEST_SEASONS = [CUR - 4, CUR - 3, CUR - 2, CUR - 1]


def american_to_prob(o):
    o = np.asarray(o, float)
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


def payout(o):  # profit per 1 unit risked
    o = np.asarray(o, float)
    return np.where(o < 0, 100 / -o, o / 100)


def fit(train):
    tr = train[train.result.notna() & (train.season >= 2021)]
    mm = Ridge(alpha=1.0).fit(tr[MARGIN_F], tr.result)
    tt = Ridge(alpha=1.0).fit(tr[TOTAL_F], tr.home_score + tr.away_score)
    # how much to trust the model vs the market: regress outcome on both
    both = np.c_[mm.predict(tr[MARGIN_F]), tr.spread_line]
    bm = Ridge(alpha=1.0, fit_intercept=False).fit(both, tr.result)
    bothT = np.c_[tt.predict(tr[TOTAL_F]), tr.total_line]
    bt = Ridge(alpha=1.0, fit_intercept=False).fit(bothT, tr.home_score + tr.away_score)
    return dict(mm=mm, tt=tt, bm=bm, bt=bt)


def predict(m, X):
    X = X.copy()
    X['model_margin'] = m['mm'].predict(X[MARGIN_F])
    X['model_total'] = m['tt'].predict(X[TOTAL_F])
    X['blend_margin'] = m['bm'].predict(np.c_[X.model_margin, X.spread_line])
    X['blend_total'] = m['bt'].predict(np.c_[X.model_total, X.total_line])
    X['p_home_cover'] = 1 - norm.cdf(X.spread_line, X.blend_margin, MARGIN_SD)
    X['p_over'] = 1 - norm.cdf(X.total_line, X.blend_total, TOTAL_SD)
    X['p_home_win'] = 1 - norm.cdf(0, X.blend_margin, MARGIN_SD)
    return X


def backtest(X):
    out = []
    for s in TEST_SEASONS:
        m = fit(X[X.season < s])
        out.append(predict(m, X[(X.season == s) & X.result.notna()]))
    return pd.concat(out)


def grade(bt):
    rows = []
    # spreads (assume -110 when juice missing)
    b = bt.copy()
    b['side_home'] = b.p_home_cover >= 0.5
    b['p_side'] = np.where(b.side_home, b.p_home_cover, 1 - b.p_home_cover)
    push = b.result == b.spread_line
    win = np.where(b.side_home, b.result > b.spread_line, b.result < b.spread_line)
    odds = np.where(b.side_home, b.home_spread_odds, b.away_spread_odds)
    odds = np.where(np.isnan(odds.astype(float)), -110, odds)
    b['ats_profit'] = np.where(push, 0, np.where(win, payout(odds), -1))
    b['ats_ev'] = b.p_side * payout(odds) - (1 - b.p_side)
    # totals
    tot = b.home_score + b.away_score
    b['side_over'] = b.p_over >= 0.5
    b['p_tside'] = np.where(b.side_over, b.p_over, 1 - b.p_over)
    tpush = tot == b.total_line
    twin = np.where(b.side_over, tot > b.total_line, tot < b.total_line)
    todds = np.where(b.side_over, b.over_odds, b.under_odds)
    todds = np.where(np.isnan(todds.astype(float)), -110, todds)
    b['tot_profit'] = np.where(tpush, 0, np.where(twin, payout(todds), -1))
    b['tot_ev'] = b.p_tside * payout(todds) - (1 - b.p_tside)
    # moneyline: bet side with positive EV vs no-vig? use actual price
    ph = b.p_home_win
    ev_h = ph * payout(b.home_moneyline) - (1 - ph)
    ev_a = (1 - ph) * payout(b.away_moneyline) - ph
    b['ml_home'] = ev_h >= ev_a
    b['ml_ev'] = np.maximum(ev_h, ev_a)
    home_won = b.result > 0
    b['ml_profit'] = np.where(b.result == 0, 0, np.where(b.ml_home,
                              np.where(home_won, payout(b.home_moneyline), -1),
                              np.where(~home_won, payout(b.away_moneyline), -1)))
    return b


def summarize(b):
    lines = []
    for mkt, ev, pr in [('Spread', 'ats_ev', 'ats_profit'), ('Total', 'tot_ev', 'tot_profit'),
                        ('Moneyline', 'ml_ev', 'ml_profit')]:
        d = b[b[ev].notna() & b[pr].notna()]
        for lo, hi in [(-9, 0), (0, 0.03), (0.03, 0.06), (0.06, 9)]:
            s = d[(d[ev] > lo) & (d[ev] <= hi)]
            if len(s):
                dec = s[s[pr] != 0]
                lines.append(dict(market=mkt, ev_bucket=f'{lo:+.0%} to {hi:+.0%}' if hi < 9 else f'>{lo:+.0%}',
                                  bets=len(s), win_rate=(dec[pr] > 0).mean(), roi=s[pr].mean()))
    return pd.DataFrame(lines)


if __name__ == '__main__':
    X = pd.read_pickle(f'{DATA}/features.pkl')
    bt = grade(backtest(X))
    bt.to_pickle(f'{DATA}/game_backtest.pkl')
    m = fit(X[X.season < CUR])
    print('margin coefs', dict(zip(MARGIN_F, m['mm'].coef_.round(2))), 'blend w', m['bm'].coef_.round(2))
    print('total coefs', dict(zip(TOTAL_F, m['tt'].coef_.round(2))), 'blend w', m['bt'].coef_.round(2))
    print('MAE margin model/market/blend:', *(bt.result - bt[c]).abs().mean().__round__(2) if False else
          [round((bt.result - bt[c]).abs().mean(), 2) for c in ['model_margin', 'spread_line', 'blend_margin']])
    print(summarize(bt).to_string())
