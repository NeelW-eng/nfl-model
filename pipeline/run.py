"""Weekly pipeline: nflverse data + injuries -> projections -> Underdog lines scored -> Top 10 and
top 4 per game -> season tracker -> dashboard data (data/board.json) and page (site/sunday_edge.html).

Runs in GitHub Actions (.github/workflows/weekly.yml) or locally:  python pipeline/run.py
"""
import json, os, sys, datetime, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, CUR, SITE
import features, model, props2, injuries, underdog, track_ud, gamelines

ODDS = os.path.join(DATA, 'odds.json')


def pick_week(games):
    """The NFL week whose games fall in the next 8 days (Thursday through Monday)."""
    g = games[(games.season == CUR) & (games.game_type == 'REG') & games.home_score.isna()].copy()
    g['kick'] = pd.to_datetime(g.gameday + ' ' + g.gametime.fillna('13:00'))
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    soon = g[(g.kick > now - pd.Timedelta(hours=4)) & (g.kick < now + pd.Timedelta(days=8))]
    return int(soon.week.mode().iloc[0]) if len(soon) else int(g.week.min())


def projections_table(proj, status):
    allow = {'passing_yards': 'df_allow_passing_yards', 'passing_tds': 'df_allow_passing_yards',
             'rushing_yards': 'df_allow_rushing_yards', 'receiving_yards': 'df_allow_receiving_yards',
             'receptions': 'df_allow_receptions', 'anytime_td': 'df_allow_tds'}
    out = {}
    for stat, d in proj.items():
        d = d[d.n_prior >= 3].sort_values('proj', ascending=False).head(15)
        rows = []
        for r in d.to_dict('records'):
            st = status.get(r['player_id'])
            x = dict(player=r['player_display_name'], team=r['team'], opp=r['opponent_team'], pos=r['position'],
                     proj=round(float(r['proj']), 2 if stat in ('passing_tds', 'anytime_td') else 1),
                     matchup=round(float(r[allow[stat]]), 2) if pd.notna(r.get(allow[stat])) else 1.0,
                     injury=st['status'] if st and st.get('flag') else None,
                     boost=round(float(r.get('inj_boost', 1.0)), 2))
            if stat in props2.YARDS:
                x.update(lo=round(float(r['q25']), 1), hi=round(float(r['q75']), 1))
            if stat == 'anytime_td':
                x['p_td'] = round(float(r['p_td']), 3)
            rows.append(x)
        out[stat] = rows
    return out


def injury_report(notes, status, teams):
    """Key players out for this week's teams, plus notable Questionable tags."""
    rows = []
    for team, lst in notes.items():
        if team in teams:
            for n in lst:
                rows.append(dict(team=team, **n))
    return sorted(rows, key=lambda r: (r['team'], r['player']))


def slim(c):
    keep = ['rank', 'game', 'game_id', 'kickoff', 'started', 'player', 'player_id', 'team', 'opp', 'pos', 'stat',
            'stat_label', 'side', 'word', 'line', 'pick', 'p', 'p_model', 'p_cons', 'n_books', 'proj', 'injury',
            'injury_note', 'signals', 'support', 'n_signals', 'in_top10']
    return {k: c.get(k) for k in keep if k in c}


def main():
    games = pd.read_csv(f'{DATA}/games.csv')
    week = int(os.environ.get('NFL_WEEK') or pick_week(games))
    print('season', CUR, 'week', week)

    X = features.build()
    X.to_pickle(f'{DATA}/features.pkl')

    status = injuries.statuses()
    proj = props2.project_week(week)
    proj, notes = injuries.adjust_projections(proj, status)
    wk_games = games[(games.season == CUR) & (games.week == week)]
    teams = set(wk_games.home_team) | set(wk_games.away_team)

    out = dict(season=CUR, week=week, generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='minutes'),
               has_odds=False, top10=[], games=[], all_lines=[], breakeven=underdog.BREAKEVEN)
    if os.path.exists(ODDS):
        odds = json.load(open(ODDS))
        out.update(odds_pulled=odds.get('pulled_at'), props_pulled=odds.get('props_pulled_at'),
                   credits_left=odds.get('credits_left'))
        cands = underdog.candidates(odds, proj, status, notes)
        lock_top, lock_game, lock_gl = track_ud.locked(CUR, week)
        top, glist = underdog.select(cands, lock_top, lock_game)
        gl = gamelines.picks(odds, X, CUR, week, notes)
        # started games keep the game-line picks that were logged before kickoff
        locked_by_game = {}
        for c in lock_gl:
            locked_by_game.setdefault(c['game_id'], []).append(c)
        for gid, cs in locked_by_game.items():
            if gid in gl:
                gl[gid]['picks'] = cs
        games_out = []
        seen = set()
        for g in glist:
            gg = dict(g, picks=[slim(c) for c in g['picks']])
            gg['lines'] = gl.get(g['game_id'], {}).get('picks', [])
            games_out.append(gg); seen.add(g['game_id'])
        for gid, x in gl.items():
            if gid not in seen:
                games_out.append(dict(game_id=gid, game=x['game'], kickoff=x['kickoff'], picks=[], lines=x['picks']))
        games_out.sort(key=lambda g: (g['kickoff'], g['game']))
        out.update(has_odds=bool(cands), lines_priced=len(cands), top10=[slim(c) for c in top], games=games_out,
                   all_lines=underdog.all_lines(cands))
        gl_flat = [c for x in gl.values() for c in x['picks']]
        if cands or gl_flat:
            track_ud.log_picks(CUR, week, top, glist, odds.get('props_pulled_at') or odds.get('pulled_at'), gl_flat)
    track_ud.grade()
    out['tracker'] = track_ud.record(CUR)

    # games this week without Underdog lines yet (shown as "lines not posted")
    have = {g['game'] for g in out['games'] if g['picks'] or g.get('lines')}
    out['schedule'] = [dict(game=f"{r.away_team} @ {r.home_team}", kickoff=f"{r.gameday} {r.gametime}",
                            has_lines=f"{r.away_team} @ {r.home_team}" in have)
                       for r in wk_games.sort_values(['gameday', 'gametime']).itertuples()]
    out['injuries'] = injury_report(notes, status, teams)
    out['injury_sources'] = ['NFL injury reports (via nflverse)', 'ESPN injury feed'] if os.path.exists(
        f'{DATA}/injuries_espn.json') else ['NFL injury reports (via nflverse)']

    m = model.fit(X[X.season < CUR])
    nxt = model.predict(m, X[(X.season == CUR) & (X.week == week)])
    out['fair_lines'] = [dict(game=f"{r.away_team} @ {r.home_team}", kickoff=f"{r.gameday.date()} {r.gametime}",
                              market_home_spread=-r.spread_line, model_home_spread=round(-r.model_margin, 1),
                              market_total=r.total_line, model_total=round(r.model_total, 1),
                              home_win_p=round(float(r.p_home_win), 3))
                         for r in nxt.itertuples() if pd.notna(r.spread_line)]
    out['projections'] = projections_table(proj, status)
    json.dump(out, open(f'{DATA}/week_picks.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o), indent=1)
    os.makedirs(SITE, exist_ok=True)
    import build_dashboard  # noqa: F401  (writes site/sunday_edge.html and data/board.json)
    print('top10', len(out['top10']), 'games', len(out['games']), 'lines', out.get('lines_priced', 0))


if __name__ == '__main__':
    main()
