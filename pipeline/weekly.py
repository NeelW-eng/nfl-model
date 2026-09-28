"""Weekly run: FanDuel lines vs. sharp consensus + model -> picks with unit sizes.

Inputs : data/fanduel_odds.json (from fetch_fanduel_odds.py on the user's Mac), nflverse data.
Output : data/week_picks.json (feeds the dashboard)
"""
import json, os, re, datetime
import pandas as pd, numpy as np
from scipy.stats import norm
import model, props

from paths import DATA, CUR, FIRST
TEAMS = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'}
SHARP = {'pinnacle': 3.0, 'lowvig': 1.5, 'betonlineag': 1.5, 'circasports': 2.0}  # consensus weights (default 1)
PROP_MKT = {'player_pass_yds': 'passing_yards', 'player_pass_tds': 'passing_tds', 'player_rush_yds': 'rushing_yards',
            'player_reception_yds': 'receiving_yards', 'player_receptions': 'receptions',
            'player_anytime_td': 'anytime_td'}
MIN_EV = 0.02


def units(ev, p, n_books, agrees):
    """Unit size from edge + certainty. Base: 1u ~ a solid edge. Capped at 3u."""
    if ev < MIN_EV:
        return 0
    u = 0.5 if ev < 0.04 else 1.0 if ev < 0.06 else 1.5 if ev < 0.09 else 2.0
    if n_books >= 5:
        u += 0.5          # well-supported consensus -> more certain the FanDuel price is off
    if agrees:
        u += 0.5          # our stats model points the same way
    if p < 0.35:
        u = min(u, 1.0)   # long shots: high variance, keep small
    return min(u, 3.0)


def devig(p1, p2):
    s = p1 + p2
    return p1 / s, p2 / s


def wmedian(vals, w):
    o = np.argsort(vals); v, w = np.asarray(vals)[o], np.asarray(w)[o]
    c = np.cumsum(w); return v[np.searchsorted(c, c[-1] / 2)]


def norm_name(n):
    n = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', n.lower())
    return re.sub(r'[^a-z]', '', n)


# ---------- game lines ----------
def game_picks(odds, feats, week):
    out = []
    m = model.fit(feats[feats.season < CUR])
    nxt = feats[(feats.season == CUR) & (feats.week == week)]
    for g in odds['games']:
        h, a = TEAMS.get(g['home_team']), TEAMS.get(g['away_team'])
        row = nxt[(nxt.home_team == h) & (nxt.away_team == a)]
        if row.empty:
            continue
        books = {b['key']: {mk['key']: mk['outcomes'] for mk in b['markets']} for b in g['bookmakers']}
        fd = books.get('fanduel')
        if not fd:
            continue
        info = dict(game=f"{a} @ {h}", home=h, away=a, kickoff=g['commence_time'])
        # --- spread: convert each book to an implied mean home margin ---
        mus, ws = [], []
        for k, mk in books.items():
            if k == 'fanduel' or 'spreads' not in mk:
                continue
            oc = {o['name']: o for o in mk['spreads']}
            if g['home_team'] not in oc or g['away_team'] not in oc:
                continue
            ph, _ = devig(*model.american_to_prob([oc[g['home_team']]['price'], oc[g['away_team']]['price']]))
            pt = oc[g['home_team']]['point']            # e.g. -3.5 = home gives 3.5
            mus.append(-pt + model.MARGIN_SD * norm.ppf(ph)); ws.append(SHARP.get(k, 1.0))
        r = row.copy(); r['spread_line'] = wmedian(mus, ws) if mus else np.nan
        # totals consensus
        tmu, tw = [], []
        for k, mk in books.items():
            if k == 'fanduel' or 'totals' not in mk:
                continue
            oc = {o['name']: o for o in mk['totals']}
            if 'Over' in oc and 'Under' in oc:
                po, _ = devig(*model.american_to_prob([oc['Over']['price'], oc['Under']['price']]))
                tmu.append(oc['Over']['point'] + model.TOTAL_SD * norm.ppf(po)); tw.append(SHARP.get(k, 1.0))
        r['total_line'] = wmedian(tmu, tw) if tmu else np.nan
        if mus and tmu:
            r = model.predict(m, r)
            fair_margin, fair_total = r.blend_margin.iloc[0], r.blend_total.iloc[0]
            model_margin, model_total = r.model_margin.iloc[0], r.model_total.iloc[0]
        else:
            continue
        info.update(fair_home_spread=round(-fair_margin, 1), fair_total=round(fair_total, 1),
                    model_home_spread=round(-model_margin, 1), model_total=round(model_total, 1),
                    n_books=len(mus))
        cand = []
        oc = {o['name']: o for o in fd.get('spreads', [])}
        for team, abbr, sign in [(g['home_team'], h, 1), (g['away_team'], a, -1)]:
            if team not in oc:
                continue
            pt, price = oc[team]['point'], oc[team]['price']
            thresh = -pt if sign == 1 else pt           # home needs margin > thresh (away: < thresh)
            p = 1 - norm.cdf(thresh, fair_margin, model.MARGIN_SD) if sign == 1 else norm.cdf(thresh, fair_margin, model.MARGIN_SD)
            agrees = (model_margin - fair_margin) * sign > 0
            cand.append(dict(market='Spread', pick=f"{abbr} {pt:+g}", price=price, p=p, agrees=agrees))
        oc = {o['name']: o for o in fd.get('totals', [])}
        for side in ['Over', 'Under']:
            if side in oc:
                pt, price = oc[side]['point'], oc[side]['price']
                p = 1 - norm.cdf(pt, fair_total, model.TOTAL_SD)
                p = p if side == 'Over' else 1 - p
                agrees = (model_total - fair_total) * (1 if side == 'Over' else -1) > 0
                cand.append(dict(market='Total', pick=f"{side} {pt:g}", price=price, p=p, agrees=agrees))
        oc = {o['name']: o for o in fd.get('h2h', [])}
        # moneyline fair prob straight from devigged consensus h2h (sharper than margin->prob)
        ph_list, pw = [], []
        for k, mk in books.items():
            if k != 'fanduel' and 'h2h' in mk:
                o2 = {o['name']: o['price'] for o in mk['h2h']}
                if g['home_team'] in o2 and g['away_team'] in o2:
                    ph_list.append(devig(*model.american_to_prob([o2[g['home_team']], o2[g['away_team']]]))[0])
                    pw.append(SHARP.get(k, 1.0))
        if ph_list:
            ph = float(np.average(ph_list, weights=pw))
            for team, abbr, p in [(g['home_team'], h, ph), (g['away_team'], a, 1 - ph)]:
                if team in oc:
                    agrees = ((model_margin - fair_margin) * (1 if abbr == h else -1)) > 0
                    cand.append(dict(market='Moneyline', pick=f"{abbr} ML", price=oc[team]['price'], p=p, agrees=agrees))
        for c in cand:
            c['ev'] = float(c['p'] * model.payout(c['price']) - (1 - c['p']))
            c['units'] = units(c['ev'], c['p'], len(mus), c['agrees'])
            c['fair_odds'] = prob_to_american(c['p'])
            c.update(info)
            c['p'] = float(c['p']); c['agrees'] = bool(c['agrees'])
            out.append(c)
    return out


def prob_to_american(p):
    p = min(max(p, 0.01), 0.99)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


# ---------- props ----------
def upcoming_projections(week):
    ps = props.load()
    g = model.pd.read_csv(f'{DATA}/games.csv')
    sch = g[(g.season == CUR) & (g.week == week)]
    opp = {**dict(zip(sch.home_team, sch.away_team)), **dict(zip(sch.away_team, sch.home_team))}
    cur = ps[ps.season == CUR]
    last = cur.sort_values('t').groupby('player_id').tail(1)
    played = set(cur[cur.week == week].team)
    last = last[last.team.isin(opp) & ~last.team.isin(played)]
    fut = last.copy()
    fut['week'] = week; fut['t'] = CUR * 100 + week
    fut['opponent_team'] = fut.team.map(opp)
    for c in ['passing_yards', 'passing_tds', 'rushing_yards', 'receiving_yards', 'receptions',
              'anytime_td', 'tds', 'rushing_tds', 'receiving_tds']:
        fut[c] = np.nan
    allp = pd.concat([ps, fut]).sort_values(['player_id', 't'])
    res = {}
    for stat in props.STATS:
        d = props.project(allp, stat)
        f = d[d.t == fut.t.iloc[0]][['player_id', 'player_display_name', 'position', 'team',
                                     'opponent_team', 'proj', 'n', 'dfac']]
        res[stat] = f
    fits = {s: props.fit_spread(props.project(ps, s), s) for s in
            ['passing_yards', 'rushing_yards', 'receiving_yards']}
    inj = pd.read_parquet(f'{DATA}/inj_{CUR}.parquet')
    inj = inj[inj.week == inj.week.max()].set_index('gsis_id').report_status.to_dict()
    return res, fits, inj


def prop_picks(odds, week):
    res, fits, inj = upcoming_projections(week)
    lookup = {s: {norm_name(n): r for n, r in zip(f.player_display_name, f.itertuples())}
              for s, f in res.items()}
    out = []
    for gid, ev in odds.get('props', {}).items():
        # collect by (market, player): fanduel offer + consensus from others
        offers = {}
        for b in ev.get('bookmakers', []):
            for mk in b['markets']:
                stat = PROP_MKT.get(mk['key'])
                if not stat:
                    continue
                for o in mk['outcomes']:
                    key = (stat, o.get('description', ''))
                    offers.setdefault(key, {}).setdefault(b['key'], []).append(o)
        for (stat, player), bk in offers.items():
            if 'fanduel' not in bk:
                continue
            pr = lookup[stat].get(norm_name(player))
            if pr is None:
                continue
            proj = float(pr.proj)
            # consensus implied median from other books
            mus = []
            for k, os_ in bk.items():
                if k == 'fanduel':
                    continue
                d = {o['name']: o for o in os_}
                if 'Over' in d and 'Under' in d and stat != 'anytime_td':
                    po, _ = devig(*model.american_to_prob([d['Over']['price'], d['Under']['price']]))
                    mus.append((d['Over']['point'], po))
                elif stat == 'anytime_td' and ('Yes' in d or 'Over' in d):
                    o = d.get('Yes') or d.get('Over')
                    mus.append((0.5, float(model.american_to_prob(o['price'])) / 1.08))
            for o in bk['fanduel']:
                side = o['name']
                if side not in ('Over', 'Under', 'Yes'):
                    continue
                line = o.get('point', 0.5)
                p_model = props.prob_over(stat, proj, line, fits.get(stat))
                p_cons = [props_consensus_p(stat, pt, po, line, fits.get(stat), proj) for pt, po in mus]
                p_cons = [x for x in p_cons if x is not None]
                # blend: consensus is sharper, model adds information props books often miss
                p_over = (0.6 * np.median(p_cons) + 0.4 * p_model) if p_cons else p_model
                p = p_over if side in ('Over', 'Yes') else 1 - p_over
                evv = float(p * model.payout(o['price']) - (1 - p))
                agrees = (p_model > 0.5) == (side in ('Over', 'Yes'))
                u = units(evv, p, len(p_cons), agrees)
                if not p_cons:
                    u = min(u, 1.0)  # no consensus to lean on -> model only, keep small
                status = inj.get(pr.player_id)
                out.append(dict(market='Prop', stat=stat, player=player, team=pr.team, opp=pr.opponent_team,
                                pick=f"{player} {'Anytime TD' if stat == 'anytime_td' else side + ' ' + format(line, 'g')} {LABEL[stat] if stat != 'anytime_td' else ''}".strip(),
                                price=o['price'], p=float(p), ev=evv, units=0 if status == 'Out' else u,
                                proj=round(proj, 2 if stat in ('passing_tds', 'anytime_td') else 1),
                                n_books=len(p_cons), agrees=bool(agrees), injury=status,
                                fair_odds=prob_to_american(p)))
    return out


LABEL = {'passing_yards': 'pass yds', 'passing_tds': 'pass TDs', 'rushing_yards': 'rush yds',
         'receiving_yards': 'rec yds', 'receptions': 'receptions', 'anytime_td': ''}


def props_consensus_p(stat, pt, p_over_at_pt, line, sdfit, proj):
    """Move another book's devigged over-prob at its point to FanDuel's point."""
    if stat == 'anytime_td':
        return p_over_at_pt
    if stat in ('passing_yards', 'rushing_yards', 'receiving_yards'):
        sd = max(np.polyval(sdfit, proj), 8)
    elif stat == 'receptions':
        sd = np.sqrt(proj * (1 + proj / 12)) + 0.3
    else:
        sd = np.sqrt(max(proj, 0.3))
    mu = pt + sd * norm.ppf(np.clip(p_over_at_pt, 0.02, 0.98))
    return float(1 - norm.cdf(line, mu, sd))


def top_projections(week, k=12):
    res, _, inj = upcoming_projections(week)
    out = {}
    for stat, f in res.items():
        f = f[f.n >= 3].sort_values('proj', ascending=False).head(k)
        out[stat] = [dict(player=r.player_display_name, team=r.team, opp=r.opponent_team,
                          proj=round(float(r.proj), 2 if stat in ('passing_tds', 'anytime_td') else 1),
                          matchup=round(float(r.dfac), 2), injury=inj.get(r.player_id))
                     for r in f.itertuples()]
    return out


def run(week):
    feats = pd.read_pickle(f'{DATA}/features.pkl')
    path = f'{DATA}/fanduel_odds.json'
    out = dict(week=week, generated=datetime.datetime.now().isoformat(timespec='minutes'),
               has_odds=os.path.exists(path), games=[], props=[])
    if out['has_odds']:
        odds = json.load(open(path))
        out['odds_pulled'] = odds['pulled_at']
        out['games'] = game_picks(odds, feats, week)
        out['props'] = prop_picks(odds, week)
    # always include model fair lines + top projections
    m = model.fit(feats[feats.season < CUR])
    nxt = model.predict(m, feats[(feats.season == CUR) & (feats.week == week)])
    out['fair_lines'] = [dict(game=f"{r.away_team} @ {r.home_team}", kickoff=f"{r.gameday.date()} {r.gametime}",
                              market_home_spread=-r.spread_line, model_home_spread=round(-r.model_margin, 1),
                              market_total=r.total_line, model_total=round(r.model_total, 1),
                              home_win_p=round(float(r.p_home_win), 3))
                         for r in nxt.itertuples()]
    out['projections'] = top_projections(week)
    json.dump(out, open(f'{DATA}/week_picks.json', 'w'), default=float, indent=1)
    return out


if __name__ == '__main__':
    import sys
    o = run(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    print('games', len(o['games']), 'props', len(o['props']))
    print(pd.DataFrame(o['fair_lines']).to_string())
    for s, v in o['projections'].items():
        print(s, [(x['player'], x['proj']) for x in v[:5]])
