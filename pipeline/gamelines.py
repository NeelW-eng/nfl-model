"""Game-line picks (spread, total, moneyline) for every game, priced from the sportsbook consensus.

Underdog's sportsbook isn't available from any odds feed, so each game is priced from 10 sportsbooks
(Pinnacle weighted 3x as the sharpest): their de-vigged lines give a fair margin, total and win chance,
nudged slightly by the stats model (it gets ~5-15% weight, its tested value against closing lines).
Each pick shows the line it's based on and the fair price: on Underdog, take it only at that price or better.
"""
import datetime
import numpy as np, pandas as pd
from scipy.stats import norm
import model
from picks import TEAMS, SHARP, implied, devig, wmedian, to_american

SHARP = dict(SHARP, pinnacle=3.0)


def ts(s):
    return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))


def consensus(g):
    """Fair home margin, fair total, home win chance and the most common posted lines."""
    mus, ws, tmu, tw, phs, pw, spreads, totals = [], [], [], [], [], [], [], []
    for b in g.get('bookmakers', []):
        mk = {m['key']: m['outcomes'] for m in b['markets']}
        wt = SHARP.get(b['key'], 1.0)
        oc = {o['name']: o for o in mk.get('spreads', [])}
        if g['home_team'] in oc and g['away_team'] in oc:
            ph, _ = devig(implied(oc[g['home_team']]['price']), implied(oc[g['away_team']]['price']))
            pt = oc[g['home_team']]['point']
            mus.append(-pt + model.MARGIN_SD * norm.ppf(ph)); ws.append(wt); spreads.append(pt)
        oc = {o['name']: o for o in mk.get('totals', [])}
        if 'Over' in oc and 'Under' in oc:
            po, _ = devig(implied(oc['Over']['price']), implied(oc['Under']['price']))
            tmu.append(oc['Over']['point'] + model.TOTAL_SD * norm.ppf(po)); tw.append(wt); totals.append(oc['Over']['point'])
        oc = {o['name']: o['price'] for o in mk.get('h2h', [])}
        if g['home_team'] in oc and g['away_team'] in oc:
            phs.append(devig(implied(oc[g['home_team']]), implied(oc[g['away_team']]))[0]); pw.append(wt)
    if not mus or not tmu or not phs:
        return None
    mode = lambda xs: float(pd.Series(xs).mode().iloc[0])
    return dict(margin=wmedian(mus, ws), total=wmedian(tmu, tw), p_home=float(np.average(phs, weights=pw)),
                home_spread=mode(spreads), total_line=mode(totals), n_books=len(mus))


def picks(odds, X, season, week, status_notes):
    """Spread, total and moneyline pick for each upcoming game this week."""
    m = model.fit(X[X.season < season])
    wk = X[(X.season == season) & (X.week == week)]
    now = datetime.datetime.now(datetime.timezone.utc)
    out = {}
    for g in odds.get('games', []):
        h, a = TEAMS.get(g['home_team']), TEAMS.get(g['away_team'])
        row = wk[(wk.home_team == h) & (wk.away_team == a)]
        if row.empty:
            continue
        c = consensus(g)
        if c is None:
            continue
        r = row.copy(); r['spread_line'] = c['margin']; r['total_line'] = c['total']
        r = model.predict(m, r).iloc[0]
        fm, ft = float(r.blend_margin), float(r.blend_total)          # consensus nudged by the model
        mod_m, mod_t = float(r.model_margin), float(r.model_total)
        game = f'{a} @ {h}'
        started = ts(g['commence_time']) <= now
        rest = float(r.rest_diff) if pd.notna(r.rest_diff) else 0.0
        qb_out = {t for t, lst in status_notes.items() if any(n['pos'] == 'QB' for n in lst)}

        def item(market, team, side, line, p, label, sig):
            return dict(market=market, game=game, game_id=g['id'], kickoff=g['commence_time'], started=started,
                        home=h, away=a, team=team, side=side, line=line, p=round(float(p), 4),
                        fair_odds=to_american(p), pick=label, n_books=c['n_books'],
                        signals=sig, support=sum(1 for s in sig if s['ok']), n_signals=len(sig))
        res = []
        # spread at the consensus line (home_spread is the home team's posted number, e.g. -3.5)
        hs = c['home_spread']
        p_home_cover = 1 - norm.cdf(-hs, fm, model.MARGIN_SD)
        for team, s, p, line in ((h, 1, p_home_cover, hs), (a, -1, 1 - p_home_cover, -hs)):
            if p >= 0.5:
                lean = s * (mod_m - c['margin'])
                sig = [dict(k='market', ok=True, txt=f"{c['n_books']} sportsbooks: fair line {team} {(-c['margin'] if s == 1 else c['margin']):+.1f}"),
                       dict(k='model', ok=lean >= 1.0, txt=f"Stats model leans {team} by {lean:+.1f} pts"),
                       dict(k='rest', ok=s * rest >= 3, txt=f"Rest edge {s * rest:+.0f} days"),
                       dict(k='injury', ok=(a if s == 1 else h) in qb_out, txt=f"{a if s == 1 else h} starting QB out")]
                res.append(item('Spread', team, 'cover', line, p, f"{team} {line:+g}", sig))
        tl = c['total_line']
        p_over = 1 - norm.cdf(tl, ft, model.TOTAL_SD)
        for side, p in (('Over', p_over), ('Under', 1 - p_over)):
            if p >= 0.5:
                lean = (mod_t - c['total']) * (1 if side == 'Over' else -1)
                wind = float(r.wind_f) if pd.notna(r.wind_f) else 0.0
                sig = [dict(k='market', ok=True, txt=f"{c['n_books']} sportsbooks: fair total {c['total']:.1f}"),
                       dict(k='model', ok=lean >= 1.5, txt=f"Stats model total {mod_t:.1f}"),
                       dict(k='weather', ok=side == 'Under' and wind >= 15, txt=f"Wind {wind:.0f} mph"),
                       dict(k='injury', ok=side == 'Under' and bool({h, a} & qb_out), txt="A starting QB is out")]
                res.append(item('Total', None, side, tl, p, f"{side} {tl:g}", sig))
        ph = c['p_home']
        for team, p in ((h, ph), (a, 1 - ph)):
            if p >= 0.5:
                sig = [dict(k='market', ok=True, txt=f"{c['n_books']} sportsbooks: {team} wins {p:.0%}"),
                       dict(k='model', ok=(mod_m if team == h else -mod_m) > 0, txt=f"Stats model margin {team} {(mod_m if team == h else -mod_m):+.1f}")]
                res.append(item('Moneyline', team, 'win', None, p, f"{team} to win", sig))
        out[g['id']] = dict(game=game, kickoff=g['commence_time'], fair_home_spread=round(-c['margin'], 1),
                            fair_total=round(c['total'], 1), picks=res)
    return out
