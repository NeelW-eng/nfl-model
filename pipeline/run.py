"""Weekly pipeline: data -> features -> models -> FanDuel comparison -> picks -> tracking -> dashboard.

Runs in GitHub Actions (see .github/workflows/weekly.yml) or locally:  python pipeline/run.py
"""
import json, os, sys, datetime, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, CUR, SITE
import features, model, props2, picks, track

ODDS = os.path.join(DATA, 'fanduel_odds.json')


def pick_week(games):
    """The week most of the next 8 days' games belong to."""
    g = games[(games.season == CUR) & (games.game_type == 'REG') & games.home_score.isna()].copy()
    g['kick'] = pd.to_datetime(g.gameday + ' ' + g.gametime.fillna('13:00'))
    now = pd.Timestamp.now(tz='America/New_York').tz_localize(None)
    soon = g[(g.kick > now - pd.Timedelta(hours=4)) & (g.kick < now + pd.Timedelta(days=8))]
    return int(soon.week.mode().iloc[0]) if len(soon) else int(g.week.min())


def projections_table(proj, inj):
    out = {}
    allow = {'passing_yards': 'df_allow_passing_yards', 'passing_tds': 'df_allow_passing_yards',
             'rushing_yards': 'df_allow_rushing_yards', 'receiving_yards': 'df_allow_receiving_yards',
             'receptions': 'df_allow_receptions', 'anytime_td': 'df_allow_tds'}
    for stat, d in proj.items():
        d = d[d.n_prior >= 3].sort_values('proj', ascending=False).head(15)
        rows = []
        for r in d.to_dict('records'):
            x = dict(player=r['player_display_name'], team=r['team'], opp=r['opponent_team'], pos=r['position'],
                     proj=round(float(r['proj']), 2 if stat in ('passing_tds', 'anytime_td') else 1),
                     matchup=round(float(r.get(allow[stat]) or 1.0), 2) if pd.notna(r.get(allow[stat])) else 1.0,
                     injury=inj.get(r['player_id']))
            if stat in props2.YARDS:
                x.update(lo=round(float(r['q25']), 1), hi=round(float(r['q75']), 1))
            if stat == 'anytime_td':
                x['p_td'] = round(float(r['p_td']), 3)
            rows.append(x)
        out[stat] = rows
    return out


def main():
    games = pd.read_csv(f'{DATA}/games.csv')
    week = int(os.environ.get('NFL_WEEK') or pick_week(games))
    print('season', CUR, 'week', week)

    X = features.build()
    X.to_pickle(f'{DATA}/features.pkl')

    # game-model backtest (walk-forward vs closing lines)
    bt = model.grade(model.backtest(X))
    summ = model.summarize(bt)
    bsum = json.load(open(f'{DATA}/backtest_summary.json')) if os.path.exists(f'{DATA}/backtest_summary.json') else {}
    bsum.update(games=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in summ.to_dict('records')],
                n_games=int(len(bt)), seasons=f'{min(model.TEST_SEASONS)}-{max(model.TEST_SEASONS)}')
    json.dump(bsum, open(f'{DATA}/backtest_summary.json', 'w'), indent=1)

    # player projections with advanced metrics
    proj = props2.project_week(week)
    inj = {}
    if os.path.exists(f'{DATA}/inj_{CUR}.parquet'):
        i = pd.read_parquet(f'{DATA}/inj_{CUR}.parquet')
        i = i[i.week == i.week.max()]
        inj = i.set_index('gsis_id').report_status.dropna().to_dict()

    out = dict(season=CUR, week=week, generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='minutes'),
               has_odds=False, picks=[], considered=0)
    if os.path.exists(ODDS):
        odds = json.load(open(ODDS))
        fresh = pd.Timestamp(odds['pulled_at']) > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=6)
        if fresh:
            cands = picks.game_candidates(odds, X, week) + picks.prop_candidates(odds, proj, inj)
            n_locked, locked = track.locked_count(CUR, week)
            sel = picks.select(cands, max_n=max(picks.MAX_PICKS - n_locked, 0))
            out['locked'] = locked
            out['all_bets'] = picks.all_bets(cands, {(c['game'], c['market'], c['pick']) for c in sel})
            out.update(has_odds=True, odds_pulled=odds['pulled_at'], props_pulled=odds.get('props_pulled_at'),
                       picks=sel, considered=len(cands))
            track.log_picks(CUR, week, sel, odds['pulled_at'])
    log = track.grade()
    out['record'] = track.record(log)

    # board: model vs market for every game this week
    m = model.fit(X[X.season < CUR])
    nxt = model.predict(m, X[(X.season == CUR) & (X.week == week)])
    out['fair_lines'] = [dict(game=f"{r.away_team} @ {r.home_team}", kickoff=f"{r.gameday.date()} {r.gametime}",
                              market_home_spread=-r.spread_line, model_home_spread=round(-r.model_margin, 1),
                              market_total=r.total_line, model_total=round(r.model_total, 1),
                              home_win_p=round(float(r.p_home_win), 3))
                         for r in nxt.itertuples() if pd.notna(r.spread_line)]
    out['projections'] = projections_table(proj, inj)
    json.dump(out, open(f'{DATA}/week_picks.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o), indent=1)
    os.makedirs(SITE, exist_ok=True)
    import build_dashboard  # noqa: F401  (writes site/sunday_edge.html)
    print('picks', len(out['picks']), 'of', out['considered'])


if __name__ == '__main__':
    main()
