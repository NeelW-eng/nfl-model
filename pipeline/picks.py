"""Pick selection: FanDuel vs devigged consensus + advanced-metric models, ranked by how many
independent signals support each bet.

A bet qualifies when
  * blended expected value at FanDuel's price >= 2%, and
  * at least 3 signals point the same way, one of which is the market (FanDuel off the consensus)
    or the model projection.
Ranked by number of supporting signals, then expected value.
"""
import json, os, re, datetime
import numpy as np, pandas as pd
from scipy.stats import norm
import model, props2
from paths import DATA

TEAMS = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'}
SHARP = {'pinnacle': 3.0, 'lowvig': 1.5, 'betonlineag': 1.5, 'circasports': 2.0}
PROP_MKT = {'player_pass_yds': 'passing_yards', 'player_pass_tds': 'passing_tds', 'player_rush_yds': 'rushing_yards',
            'player_reception_yds': 'receiving_yards', 'player_receptions': 'receptions',
            'player_anytime_td': 'anytime_td'}
LABEL = {'passing_yards': 'pass yds', 'passing_tds': 'pass TDs', 'rushing_yards': 'rush yds',
         'receiving_yards': 'rec yds', 'receptions': 'receptions', 'anytime_td': 'anytime TD'}
MIN_EV, MIN_SUPPORT = 0.02, 3


# ---------------------------------------------------------------- helpers
def payout(o):
    return float(model.payout(o))


def implied(o):
    return float(model.american_to_prob(o))


def devig(p1, p2):
    s = p1 + p2
    return p1 / s, p2 / s


def wmedian(vals, w):
    o = np.argsort(vals); v, w = np.asarray(vals)[o], np.asarray(w)[o]
    c = np.cumsum(w); return float(v[np.searchsorted(c, c[-1] / 2)])


def norm_name(n):
    n = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b\.?', '', str(n).lower())
    return re.sub(r'[^a-z]', '', n)


def to_american(p):
    p = min(max(p, 0.01), 0.99)
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


# ---------------------------------------------------------------- game lines
def game_candidates(odds, feats, week):
    m = model.fit(feats[feats.season < feats.season.max()])
    nxt = feats[(feats.season == feats.season.max()) & (feats.week == week)]
    out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for g in odds.get('games', []):
        if datetime.datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')) <= now:
            continue  # already kicked off
        h, a = TEAMS.get(g['home_team']), TEAMS.get(g['away_team'])
        row = nxt[(nxt.home_team == h) & (nxt.away_team == a)]
        if row.empty:
            continue
        books = {b['key']: {mk['key']: mk['outcomes'] for mk in b['markets']} for b in g['bookmakers']}
        fd = books.get('fanduel')
        if not fd:
            continue
        mus, ws, tmu, tw, phs, pw, cons_spreads = [], [], [], [], [], [], []
        for k, mk in books.items():
            if k == 'fanduel':
                continue
            wt = SHARP.get(k, 1.0)
            oc = {o['name']: o for o in mk.get('spreads', [])}
            if g['home_team'] in oc and g['away_team'] in oc:
                ph, _ = devig(implied(oc[g['home_team']]['price']), implied(oc[g['away_team']]['price']))
                pt = oc[g['home_team']]['point']
                mus.append(-pt + model.MARGIN_SD * norm.ppf(ph)); ws.append(wt); cons_spreads.append(pt)
            oc = {o['name']: o for o in mk.get('totals', [])}
            if 'Over' in oc and 'Under' in oc:
                po, _ = devig(implied(oc['Over']['price']), implied(oc['Under']['price']))
                tmu.append(oc['Over']['point'] + model.TOTAL_SD * norm.ppf(po)); tw.append(wt)
            oc = {o['name']: o['price'] for o in mk.get('h2h', [])}
            if g['home_team'] in oc and g['away_team'] in oc:
                phs.append(devig(implied(oc[g['home_team']]), implied(oc[g['away_team']]))[0]); pw.append(wt)
        if not mus or not tmu:
            continue
        r = row.copy()
        r['spread_line'] = wmedian(mus, ws); r['total_line'] = wmedian(tmu, tw)
        r = model.predict(m, r).iloc[0]
        fair_m, fair_t = r.blend_margin, r.blend_total
        mod_m, mod_t = r.model_margin, r.model_total
        cons_pt = float(np.median(cons_spreads))
        base = dict(game=f"{a} @ {h}", home=h, away=a, kickoff=g['commence_time'], n_books=len(mus),
                    fair_home_spread=round(-fair_m, 1), fair_total=round(fair_t, 1),
                    model_home_spread=round(-mod_m, 1), model_total=round(mod_t, 1))
        rest = float(r.rest_diff) if pd.notna(r.rest_diff) else 0.0
        wind = float(r.wind_f) if pd.notna(r.wind_f) else 0.0

        def add(market, pick, price, p, side_sign, extra_signals, point=None, team=None):
            ev = p * payout(price) - (1 - p)
            sig = []
            cons_p = p  # fair prob is consensus-driven
            sig.append(dict(k='market', ok=ev > 0,
                            txt=f"FanDuel {price:+d} vs fair {to_american(cons_p):+d} ({len(mus)} books)"))
            sig += extra_signals
            out.append(dict(base, market=market, pick=pick, price=int(price), p=float(p), ev=float(ev),
                            fair_odds=to_american(p), signals=sig, point=point, team=team))

        # spreads
        oc = {o['name']: o for o in fd.get('spreads', [])}
        for team, abbr, s in [(g['home_team'], h, 1), (g['away_team'], a, -1)]:
            if team not in oc:
                continue
            pt, price = oc[team]['point'], oc[team]['price']
            thr = -pt if s == 1 else pt
            p = 1 - norm.cdf(thr, fair_m, model.MARGIN_SD) if s == 1 else norm.cdf(thr, fair_m, model.MARGIN_SD)
            my_cons = cons_pt if s == 1 else -cons_pt
            key = any((my_cons < kn <= pt) or (my_cons <= -kn < pt) for kn in (3, 7)) if pt > my_cons else False
            adv = s * np.nanmean([r.m_early / 0.1, r.m_expl / 0.03, -r.m_press / 0.05])
            sig = [
                dict(k='model', ok=s * (mod_m - fair_m) >= 1.0,
                     txt=f"Stats model: {abbr} {s * (mod_m - fair_m):+.1f} pts vs market"),
                dict(k='keynum', ok=bool(key), txt=f"FanDuel {pt:+g} crosses a key number vs consensus {my_cons:+g}"),
                dict(k='rest', ok=s * rest >= 3, txt=f"Rest edge {s * rest:+.0f} days"),
                dict(k='advanced', ok=bool(adv > 0.5),
                     txt="Early-down EPA, explosive plays and pressure matchup favor " + abbr),
            ]
            add('Spread', f"{abbr} {pt:+g}", price, p, s, sig, point=pt, team=abbr)
        # totals
        oc = {o['name']: o for o in fd.get('totals', [])}
        for side, s in [('Over', 1), ('Under', -1)]:
            if side not in oc:
                continue
            pt, price = oc[side]['point'], oc[side]['price']
            p = 1 - norm.cdf(pt, fair_t, model.TOTAL_SD)
            p = p if s == 1 else 1 - p
            pace = s * np.nanmean([(r.t_expl - 0.36) / 0.03, (r.t_plays - 250) / 5, -(r.t_press - 1.0) / 0.1])
            sig = [
                dict(k='model', ok=s * (mod_t - fair_t) >= 1.5, txt=f"Stats model total {mod_t:.1f} vs market {fair_t:.1f}"),
                dict(k='weather', ok=(s == -1 and wind >= 15), txt=f"Wind {wind:.0f} mph"),
                dict(k='advanced', ok=bool(pace > 0.5), txt="Pace, explosive-play and pressure rates point " + side.lower()),
            ]
            add('Total', f"{side} {pt:g}", price, p, s, sig, point=pt)
        # moneyline
        oc = {o['name']: o for o in fd.get('h2h', [])}
        if phs:
            ph = float(np.average(phs, weights=pw))
            for team, abbr, p, s in [(g['home_team'], h, ph, 1), (g['away_team'], a, 1 - ph, -1)]:
                if team in oc:
                    adv = s * np.nanmean([r.m_early / 0.1, r.m_expl / 0.03, -r.m_press / 0.05])
                    sig = [dict(k='model', ok=s * (mod_m - fair_m) >= 1.0,
                                txt=f"Stats model: {abbr} {s * (mod_m - fair_m):+.1f} pts vs market"),
                           dict(k='rest', ok=s * rest >= 3, txt=f"Rest edge {s * rest:+.0f} days"),
                           dict(k='advanced', ok=bool(adv > 0.5),
                                txt="Early-down EPA, explosive plays and pressure matchup favor " + abbr)]
                    add('Moneyline', f"{abbr} ML", oc[team]['price'], p, s, sig, team=abbr)
    return out


# ---------------------------------------------------------------- props
def prop_signals(stat, r, over):
    """Evidence beyond the price, each checked in the bet's direction (over=+1 / under=-1)."""
    s = over
    S = []
    f = lambda c: (float(r[c]) if c in r and pd.notna(r[c]) else np.nan)
    # role trend
    use = {'receiving_yards': 'target_share', 'receptions': 'target_share', 'rushing_yards': 'carry_share',
           'anytime_td': 'rz_share'}.get(stat)
    if use and use != 'rz_share':
        last, avg = f('last_' + use), f('u_' + use)
        if pd.notna(last) and pd.notna(avg) and avg > 0:
            ch = (last - avg) / avg
            S.append(dict(k='role', ok=s * ch >= 0.10, txt=f"{use.replace('_', ' ').title()} last game {last:.0%} vs {avg:.0%} avg"))
    snap_l, snap_a = f('last_offense_pct'), f('u_offense_pct')
    if pd.notna(snap_l) and pd.notna(snap_a):
        S.append(dict(k='snaps', ok=s * (snap_l - snap_a) >= 0.07, txt=f"Snap share {snap_l:.0%} last game vs {snap_a:.0%} avg"))
    # matchup
    allow = {'receiving_yards': 'df_allow_receiving_yards', 'receptions': 'df_allow_receptions',
             'rushing_yards': 'df_allow_rushing_yards', 'passing_yards': 'df_allow_passing_yards',
             'passing_tds': 'df_allow_passing_yards', 'anytime_td': 'df_allow_tds'}[stat]
    a = f(allow)
    if pd.notna(a):
        S.append(dict(k='matchup', ok=s * (a - 1) >= 0.08,
                      txt=f"{r['opponent_team']} allows {a - 1:+.0%} {LABEL[stat] if stat != 'anytime_td' else 'TDs'} to {r['position']}s vs avg"))
    # game script
    ip = f('implied_pts')
    if pd.notna(ip):
        if stat == 'rushing_yards':
            sp = f('spread_for')
            S.append(dict(k='script', ok=(s == 1 and sp >= 3.5) or (s == -1 and sp <= -3.5),
                          txt=f"{r['team']} {'favored' if sp > 0 else 'underdog'} by {abs(sp):g}"))
        else:
            S.append(dict(k='script', ok=s * (ip - 22.5) >= 2.5, txt=f"{r['team']} implied team total {ip:.1f}"))
    # efficiency (Next Gen Stats / PFR)
    if stat in ('receiving_yards', 'receptions'):
        sep, ref = f('f_ngs_rec_avg_separation'), f('ref_f_ngs_rec_avg_separation')
        if pd.notna(sep) and pd.notna(ref):
            S.append(dict(k='ngs', ok=s * (sep - ref) >= 0.3, txt=f"NGS separation {sep:.1f} yds vs {ref:.1f} {r['position']} median"))
    if stat == 'rushing_yards':
        ry = f('f_ngs_rus_rush_yards_over_expected_per_att')
        if pd.notna(ry):
            S.append(dict(k='ngs', ok=s * ry >= 0.3, txt=f"NGS rush yds over expected {ry:+.2f}/carry"))
    if stat in ('passing_yards', 'passing_tds'):
        cp = f('f_ngs_pas_completion_percentage_above_expectation')
        pr = f('df_pressure_pct')
        if pd.notna(cp):
            S.append(dict(k='ngs', ok=s * cp >= 1.5, txt=f"NGS CPOE {cp:+.1f}%"))
        if pd.notna(pr):
            S.append(dict(k='pressure', ok=s * (0.22 - pr) >= 0.03, txt=f"{r['opponent_team']} pressure rate {pr:.0%}"))
    if stat == 'anytime_td':
        rz = f('u_rz_share')
        if pd.notna(rz):
            S.append(dict(k='redzone', ok=s * (rz - 0.12) >= 0.05, txt=f"Red-zone opportunity share {rz:.0%}"))
    return S


def prop_candidates(odds, proj, injuries):
    lookup = {s: {norm_name(n): row for n, row in zip(d.player_display_name, d.to_dict('records'))}
              for s, d in proj.items()}
    out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for gid, ev in odds.get('props', {}).items():
        ct = ev.get('commence_time')
        if ct and datetime.datetime.fromisoformat(ct.replace('Z', '+00:00')) <= now:
            continue  # already kicked off
        offers = {}
        for b in ev.get('bookmakers', []):
            for mk in b['markets']:
                stat = PROP_MKT.get(mk['key'])
                if stat:
                    for o in mk['outcomes']:
                        offers.setdefault((stat, o.get('description', '')), {}).setdefault(b['key'], []).append(o)
        for (stat, player), bk in offers.items():
            if 'fanduel' not in bk or stat not in lookup:
                continue
            r = lookup[stat].get(norm_name(player))
            if r is None:
                continue
            status = injuries.get(r['player_id'])
            if status in ('Out', 'Doubtful'):
                continue
            # consensus: other books' devigged over-prob, moved to FanDuel's line with the model's spread
            others = []
            for k, os_ in bk.items():
                if k == 'fanduel':
                    continue
                d = {o['name']: o for o in os_}
                if stat == 'anytime_td':
                    o = d.get('Yes') or d.get('Over')
                    if o:
                        others.append((0.5, implied(o['price']) / 1.07))
                elif 'Over' in d and 'Under' in d:
                    others.append((d['Over'].get('point'), devig(implied(d['Over']['price']), implied(d['Under']['price']))[0]))
            for o in bk['fanduel']:
                side = o['name']
                if side not in ('Over', 'Under', 'Yes'):
                    continue
                line = o.get('point', 0.5)
                p_model_over = props2.prob_over(stat, r, line)
                cons = [consensus_at(stat, r, pt, po, line) for pt, po in others if pt is not None]
                cons = [c for c in cons if c is not None]
                p_over = 0.6 * float(np.median(cons)) + 0.4 * p_model_over if cons else p_model_over
                s = 1 if side in ('Over', 'Yes') else -1
                p = p_over if s == 1 else 1 - p_over
                pm = p_model_over if s == 1 else 1 - p_model_over
                evv = p * payout(o['price']) - (1 - p)
                be = implied(o['price'])
                sig = []
                if cons:
                    pc = float(np.median(cons)) if s == 1 else 1 - float(np.median(cons))
                    sig.append(dict(k='market', ok=pc > be, txt=f"{len(cons)} other books price this at {to_american(pc):+d}"))
                proj_txt = (f"Projection {r['proj']:.1f} vs line {line:g} ({pm:.0%} to hit)" if stat != 'anytime_td'
                            else f"Model TD chance {pm:.0%} vs {be:.0%} priced in")
                sig.append(dict(k='model', ok=pm > be + 0.02, txt=proj_txt))
                sig += prop_signals(stat, r, s)
                name = f"{player} {'Anytime TD' if stat == 'anytime_td' else side + ' ' + format(line, 'g') + ' ' + LABEL[stat]}"
                out.append(dict(market='Prop', stat=stat, pick=name, player=player, team=r['team'],
                                opp=r['opponent_team'], game=f"{r['team']} vs {r['opponent_team']}", price=int(o['price']),
                                p=float(p), ev=float(evv), fair_odds=to_american(p), proj=round(float(r['proj']), 2),
                                line=line, side=side, n_books=len(cons), injury=status, signals=sig,
                                player_id=r['player_id'], kickoff=ct))
    return out


def consensus_at(stat, r, pt, p_over_at_pt, line):
    if stat == 'anytime_td':
        return p_over_at_pt
    # find the mean that reproduces the other book's price with this player's own distribution, then re-price
    lo, hi = -50.0, 500.0
    base = dict(r)
    for _ in range(40):
        mid = (lo + hi) / 2
        rr = shift_proj(base, stat, mid)
        if props2.prob_over(stat, rr, pt) > p_over_at_pt:
            hi = mid
        else:
            lo = mid
    return props2.prob_over(stat, shift_proj(base, stat, (lo + hi) / 2), line)


def shift_proj(r, stat, new_center):
    rr = dict(r)
    if stat in props2.YARDS:
        d = new_center - r['q50']
        rr.update(q25=r['q25'] + d, q50=new_center, q75=r['q75'] + d)
    else:
        rr['proj'] = max(new_center, 0.01)
    return rr


# ---------------------------------------------------------------- selection
MAX_PICKS = 20  # per NFL week, across the Thursday and Sunday boards


def confidence(c):
    """0-100 score: how strongly the data backs this bet.
    40% size of the edge (maxes out at 8% EV), 40% share of signals that agree,
    20% depth of the market comparison (maxes out at 8 other books)."""
    edge = min(max(c['ev'], 0) / 0.08, 1.0)
    sup = c['support'] / max(c['n_signals'], 1)
    depth = min(c['n_books'] / 8, 1.0)
    score = 100 * (0.40 * edge + 0.40 * sup + 0.20 * depth)
    if c.get('injury') == 'Questionable':
        score -= 10
    return round(score, 1)


TIERS = [(85, 'Top', 3.0), (75, 'Strong', 2.0), (60, 'Solid', 1.5), (50, 'Standard', 1.0), (0, 'Lean', 0.5)]


def size(score, p, injury=None):
    tier, u = next((t, u) for cut, t, u in TIERS if score >= cut)
    if p < 0.35:
        u = min(u, 1.0)       # long shots: high variance
    if injury == 'Questionable':
        u = min(u, 0.5)
    return tier, u


def select(cands, max_n=MAX_PICKS):
    for c in cands:
        c['support'] = sum(1 for s in c['signals'] if s['ok'])
        c['n_signals'] = len(c['signals'])
        anchor = any(s['ok'] and s['k'] in ('market', 'model') for s in c['signals'])
        c['qualifies'] = bool(c['ev'] >= MIN_EV and c['support'] >= MIN_SUPPORT and anchor)
        c['confidence'] = confidence(c)
        c['tier'], c['units'] = size(c['confidence'], c['p'], c.get('injury')) if c['qualifies'] else (None, 0)
    q = sorted((c for c in cands if c['qualifies']), key=lambda c: (-c['confidence'], -c['ev']))
    # one bet per player-stat and per game market (spread and moneyline on a game count as one side bet)
    seen, picks = set(), []
    for c in q:
        key = (c.get('player_id'), c.get('stat')) if c['market'] == 'Prop' else (c['game'], c['market'] if c['market'] == 'Total' else 'side')
        if key in seen:
            continue
        seen.add(key); picks.append(c)
        if len(picks) >= max_n:
            break
    for i, c in enumerate(picks, 1):
        c['rank'] = i
    return picks


def next_week(games):
    s = games.season.max()
    g = games[(games.season == s) & (games.game_type == 'REG')]
    open_ = g[g.home_score.isna()]
    return int(s), int(open_.week.min()) if len(open_) else int(g.week.max())
