"""Underdog pick'em: score every standard Underdog line, then choose the week's Top 10 and each game's top 4.

Win chance for each Higher/Lower pick
  * consensus: every other sportsbook's de-vigged over/under price for that player and stat, moved to
    Underdog's line using the player's own outcome distribution (median across books)
  * model: the advanced-stat projection (usage x efficiency, Next Gen Stats, PFR, matchup, game script,
    injury adjustments) turned into a probability at Underdog's line
  * blended by blend(): consensus first, moved toward the model by up to 40%, less when they disagree;
    with no sportsbook pricing the line, the model is shrunk halfway to 50% and the pick can't make the Top 10.
Underdog's standard picks all pay the same multiplier, so ranking is by win chance alone. Lines with
non-standard multipliers (Underdog's "_alternate" markets) are skipped.
"""
import datetime
import numpy as np, pandas as pd
import props2
from picks import TEAMS, PROP_MKT, LABEL, implied, devig, norm_name, to_american, consensus_at, prop_signals

UD = 'underdog'
UD_LABEL = {'passing_yards': 'Pass Yards', 'passing_tds': 'Pass TDs', 'rushing_yards': 'Rush Yards',
            'receiving_yards': 'Receiving Yards', 'receptions': 'Receptions'}
TOP_N, PER_GAME_TOP10, GAME_N = 10, 2, 4
MIN_BOOKS_TOP10, MIN_HISTORY_TOP10 = 3, 4
BREAKEVEN = 0.54  # per-pick hit rate Underdog standard entries need (3.5x for 2 picks ... 120x for 8)


def ts(s):
    return datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))


def blend(p_cons, p_model):
    """Market-anchored probability of the Higher side.
    With sportsbook prices, start from their consensus and move toward the model by up to 40%, less the more
    the two disagree: a model at 95% against books at 52% almost always means the books know about a role
    change (injury, depth chart) the model hasn't seen, so the model is nearly ignored there.
    Without sportsbook prices the model is shrunk halfway to 50%."""
    from scipy.stats import norm
    pm = min(max(p_model, 0.02), 0.98)
    if p_cons is None:
        return 0.5 + 0.5 * (pm - 0.5)
    pc = min(max(p_cons, 0.02), 0.98)
    d = abs(norm.ppf(pm) - norm.ppf(pc))
    w = 0.4 * float(np.exp(-(d / 0.8) ** 2))
    return pc + w * (pm - pc)


def candidates(odds, proj, status, inj_notes):
    lookup = {s: {norm_name(n): row for n, row in zip(d.player_display_name, d.to_dict('records'))}
              for s, d in proj.items()}
    now = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for gid, ev in odds.get('props', {}).items():
        kick = ev.get('commence_time')
        if not kick:
            continue
        h, a = TEAMS.get(ev.get('home_team')), TEAMS.get(ev.get('away_team'))
        game = f'{a} @ {h}'
        started = ts(kick) <= now
        offers = {}
        for b in ev.get('bookmakers', []):
            for mk in b['markets']:
                stat = PROP_MKT.get(mk['key'])          # '_alternate' keys are not in PROP_MKT, so skipped
                if not stat:
                    continue
                for o in mk['outcomes']:
                    offers.setdefault((stat, o.get('description', '')), {}).setdefault(b['key'], []).append(o)
        for (stat, player), bk in offers.items():
            if UD not in bk or stat not in lookup:
                continue
            ud = {o['name']: o for o in bk[UD]}
            line = (ud.get('Over') or ud.get('Under') or {}).get('point')
            if line is None:
                continue
            r = lookup[stat].get(norm_name(player))
            st = None
            if r is not None:
                st = status.get(r['player_id'])
            if r is None:
                continue                                 # out (removed by injury adjustment) or no history
            others = []
            for k, os_ in bk.items():
                if k == UD:
                    continue
                d = {o['name']: o for o in os_}
                if 'Over' in d and 'Under' in d and d['Over'].get('point') is not None:
                    others.append((d['Over']['point'], devig(implied(d['Over']['price']), implied(d['Under']['price']))[0]))
            p_model = props2.prob_over(stat, r, line)
            cons = [c for c in (consensus_at(stat, r, pt, po, line) for pt, po in others) if c is not None]
            p_cons = float(np.median(cons)) if cons else None
            p_over = blend(p_cons, p_model)
            for side, s in (('Over', 1), ('Under', -1)):
                if side not in ud:
                    continue
                p = p_over if s == 1 else 1 - p_over
                if p < 0.5:
                    continue
                pm = p_model if s == 1 else 1 - p_model
                sig = []
                if cons:
                    pc = p_cons if s == 1 else 1 - p_cons
                    lines_ = sorted({pt for pt, _ in others})
                    rng = f"{lines_[0]:g}" if len(lines_) == 1 else f"{lines_[0]:g}–{lines_[-1]:g}"
                    sig.append(dict(k='market', ok=pc > 0.53, txt=f"{len(cons)} sportsbooks (lines {rng}) put this at {pc:.0%}"))
                sig.append(dict(k='model', ok=pm > 0.53, txt=f"Projection {r['proj']:.1f} vs line {line:g} ({pm:.0%})"))
                sig += prop_signals(stat, r, s)
                boost = float(r.get('inj_boost', 1.0) or 1.0)
                team_notes = inj_notes.get(r['team'], [])
                if team_notes and abs(boost - 1) > 0.01:
                    who = ', '.join(n['player'] for n in team_notes[:2])
                    sig.append(dict(k='injury', ok=(boost > 1) == (s == 1),
                                    txt=f"{who} out: projection {'up' if boost > 1 else 'down'} {abs(boost - 1):.0%}"))
                if r.get('qb_out'):
                    sig.append(dict(k='injury', ok=s == -1, txt=f"{r['team']} starting QB out"))
                word = 'Higher' if s == 1 else 'Lower'
                out.append(dict(
                    game=game, game_id=gid, kickoff=kick, started=started, home=h, away=a,
                    player=r['player_display_name'], player_id=r['player_id'], team=r['team'], opp=r['opponent_team'],
                    pos=r['position'], stat=stat, stat_label=UD_LABEL[stat], side=side, word=word, line=float(line),
                    pick=f"{r['player_display_name']} {word} {line:g} {UD_LABEL[stat]}",
                    p=round(float(p), 4), p_model=round(float(pm), 4),
                    p_cons=None if p_cons is None else round(float(p_cons if s == 1 else 1 - p_cons), 4),
                    n_books=len(cons), proj=round(float(r['proj']), 2), history=int(r.get('n_prior') or 0),
                    injury=st['status'] if st and st.get('flag') else None,
                    injury_note=st.get('note') if st and st.get('flag') else None,
                    signals=sig, support=sum(1 for x in sig if x['ok']), n_signals=len(sig)))
    return out


def top10_eligible(c):
    """Most confident: enough sportsbooks, enough player history, not on the injury report, and the
    sportsbooks and the model both lean the same way as the pick."""
    return (not c['injury'] and c['n_books'] >= MIN_BOOKS_TOP10 and c['history'] >= MIN_HISTORY_TOP10
            and (c['p_cons'] or 0) >= 0.51 and c['p_model'] >= 0.52)


def rank_key(c):
    return (-c['p'], -c['support'], -c['n_books'])


def select(cands, locked_top10=(), locked_game=()):
    """Top 10 (<=2 per game, 1 per player) and top 4 per game (1 per player).
    Picks whose game has started stay as they were logged (locked_*); open slots are refilled."""
    open_c = [c for c in cands if not c['started']]
    started_games = {c['game_id'] for c in locked_top10} | {c['game_id'] for c in locked_game}
    # Top 10
    top = list(locked_top10)
    per_game = {}
    players = {c['player_id'] for c in top}
    for c in top:
        per_game[c['game_id']] = per_game.get(c['game_id'], 0) + 1
    for c in sorted((c for c in open_c if top10_eligible(c)), key=rank_key):
        if len(top) >= TOP_N:
            break
        if c['player_id'] in players or per_game.get(c['game_id'], 0) >= PER_GAME_TOP10:
            continue
        top.append(c); players.add(c['player_id']); per_game[c['game_id']] = per_game.get(c['game_id'], 0) + 1
    top.sort(key=rank_key)
    for i, c in enumerate(top, 1):
        c['rank'] = i
    # top 4 per game
    games = {}
    for c in locked_game:
        games.setdefault(c['game_id'], []).append(c)
    by_game = {}
    for c in open_c:
        by_game.setdefault(c['game_id'], []).append(c)
    for gid, cs in by_game.items():
        if gid in games:
            continue
        chosen, seen = [], set()
        for c in sorted(cs, key=rank_key):
            if c['player_id'] in seen:
                continue
            chosen.append(c); seen.add(c['player_id'])
            if len(chosen) >= GAME_N:
                break
        games[gid] = chosen
    top_ids = {(c['player_id'], c['stat'], c['side'], c['line']) for c in top}
    game_list = []
    for gid, cs in games.items():
        cs = sorted(cs, key=rank_key)
        for c in cs:
            c['in_top10'] = (c['player_id'], c['stat'], c['side'], c['line']) in top_ids
        game_list.append(dict(game_id=gid, game=cs[0]['game'], kickoff=cs[0]['kickoff'], picks=cs))
    game_list.sort(key=lambda g: (g['kickoff'], g['game']))
    return top, game_list


def all_lines(cands):
    return [dict(pick=c['pick'], game=c['game'], stat=c['stat_label'], p=c['p'], n_books=c['n_books'],
                 injury=c['injury'], started=c['started'])
            for c in sorted(cands, key=rank_key)]
