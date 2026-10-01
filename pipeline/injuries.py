"""Injury status from credible sources, merged per player (keyed by NFL gsis id).

Sources
  1. Official NFL injury reports (practice participation and game status), published by the teams through
     the league and mirrored by nflverse (data/inj_<season>.parquet).
  2. ESPN's injury feed (site.api.espn.com), updated through the week with designations, injured reserve,
     and notes from team reports. Saved to data/injuries_espn.json by download_data.py.

When both have a player, the more severe current designation wins, so an ESPN "Out" or injured-reserve
listing is never masked by an older "Questionable" on the official report.
"""
import json, os
import pandas as pd
from paths import DATA, CUR

OUT_STATUSES = {'Out', 'Doubtful', 'Injured Reserve', 'Suspended', 'Physically Unable to Perform', 'PUP',
                'Non-Football Injury', 'Reserve/Commissioner Exempt', 'Inactive'}
FLAG_STATUSES = {'Questionable', 'Day-To-Day'}
SEVERITY = {'Inactive': 6, 'Injured Reserve': 6, 'Suspended': 6, 'Physically Unable to Perform': 6, 'PUP': 6,
            'Non-Football Injury': 6, 'Reserve/Commissioner Exempt': 6, 'Out': 5, 'Doubtful': 4,
            'Questionable': 3, 'Day-To-Day': 2, 'Active': 0}


def espn_feed():
    p = os.path.join(DATA, 'injuries_espn.json')
    if not os.path.exists(p):
        return []
    try:
        d = json.load(open(p))
    except Exception:
        return []
    ids = pd.read_parquet(f'{DATA}/players.parquet', columns=['gsis_id', 'espn_id']).dropna()
    esp = dict(zip(ids.espn_id.astype(str), ids.gsis_id))
    rows = []
    for team in d.get('injuries', []):
        for e in team.get('injuries', []):
            ath = e.get('athlete', {}) or {}
            aid = str(ath.get('id') or (ath.get('links') or [{}])[0].get('href', '').split('/id/')[-1].split('/')[0])
            gsis = esp.get(aid)
            if not gsis:
                continue
            rows.append(dict(gsis_id=gsis, name=ath.get('displayName'), status=e.get('status'),
                             note=e.get('shortComment') or '', date=e.get('date'), source='ESPN'))
    return rows


def official_report():
    p = f'{DATA}/inj_{CUR}.parquet'
    if not os.path.exists(p):
        return []
    i = pd.read_parquet(p)
    i = i[i.week == i.week.max()]
    rows = []
    for r in i.itertuples():
        st = r.report_status if isinstance(r.report_status, str) else None
        if not st:
            continue
        inj = r.report_primary_injury if isinstance(r.report_primary_injury, str) else ''
        rows.append(dict(gsis_id=r.gsis_id, name=r.full_name, status=st, note=(inj + ' - ' if inj else '') + 'NFL injury report',
                         date=None, source='NFL injury report'))
    return rows


def statuses():
    """{gsis_id: {'status', 'note', 'source', 'out': bool, 'flag': bool}} using the most severe current listing."""
    best = {}
    for r in official_report() + espn_feed():
        s = r['status']
        if s not in SEVERITY:
            continue
        cur = best.get(r['gsis_id'])
        if cur is None or SEVERITY[s] > SEVERITY[cur['status']]:
            best[r['gsis_id']] = r
    for r in best.values():
        r['out'] = r['status'] in OUT_STATUSES
        r['flag'] = r['status'] in FLAG_STATUSES
    return best


def adjust_projections(proj, status):
    """Remove players who are out and hand their targets/carries to active teammates.

    Freed target share is spread over active pass-catchers in proportion to their own share (70% of it;
    the rest goes to depth players and play-calling changes). Same for carries among running backs.
    If a team's starting quarterback is out, its pass-catchers' projections are cut 12%.
    Returns (proj, notes) where notes[(team)] lists who is out and the boost applied.
    """
    notes = {}
    out_ids = {k for k, v in status.items() if v['out']}
    base = proj.get('receiving_yards')
    if base is None:
        return proj, notes
    # usage reference from any table (same player rows)
    ref = pd.concat([d[['player_id', 'player_display_name', 'team', 'position', 'u_target_share', 'u_carry_share',
                        'u_attempts', 'u_offense_pct']] for d in proj.values()]).drop_duplicates('player_id')
    ref['out'] = ref.player_id.isin(out_ids)
    mult_rec, mult_rush, qb_out = {}, {}, set()
    for team, t in ref.groupby('team'):
        gone = t[t.out]
        if gone.empty:
            continue
        act = t[~t.out]
        freed_t = gone.u_target_share.fillna(0).sum()
        act_t = act.u_target_share.fillna(0).sum()
        if freed_t > 0.04 and act_t > 0:
            for pid in act.player_id:
                mult_rec[pid] = min(1 + 0.7 * freed_t / act_t, 1.5)
        rb_g = gone[gone.position == 'RB'].u_carry_share.fillna(0).sum()
        rb_a = act[act.position == 'RB'].u_carry_share.fillna(0).sum()
        if rb_g > 0.10 and rb_a > 0:
            for pid in act[act.position == 'RB'].player_id:
                mult_rush[pid] = min(1 + 0.7 * rb_g / rb_a, 1.6)
        if ((gone.position == 'QB') & (gone.u_attempts.fillna(0) > 20)).any():
            qb_out.add(team)
        key = gone[(gone.u_target_share.fillna(0) > 0.08) | (gone.u_carry_share.fillna(0) > 0.2) |
                   (gone.u_attempts.fillna(0) > 20)]
        if len(key):
            notes[team] = [dict(player=r.player_display_name, pos=r.position, status=status[r.player_id]['status'],
                                source=status[r.player_id]['source']) for r in key.itertuples()]
    for stat, d in proj.items():
        d = d[~d.player_id.isin(out_ids)].copy()
        cols = [c for c in ('proj', 'q25', 'q50', 'q75') if c in d]
        m = pd.Series(1.0, index=d.index)
        if stat in ('receiving_yards', 'receptions'):
            m *= d.player_id.map(mult_rec).fillna(1.0)
            m *= d.team.map(lambda t: 0.88 if t in qb_out else 1.0)
        if stat == 'rushing_yards':
            m *= d.player_id.map(mult_rush).fillna(1.0)
        if stat == 'anytime_td':
            boost = d.player_id.map(mult_rec).fillna(1.0) * d.player_id.map(mult_rush).fillna(1.0)
            d['p_td'] = 1 - (1 - d.p_td) ** boost.clip(upper=1.4)
        for c in cols:
            d[c] = d[c] * m
        d['inj_boost'] = m.round(3)
        d['qb_out'] = d.team.isin(qb_out)
        proj[stat] = d
    return proj, notes
