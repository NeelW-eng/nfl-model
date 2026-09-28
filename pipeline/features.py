"""Team-level features built only from games played BEFORE each game (no leakage)."""
import pandas as pd, numpy as np

from paths import DATA, CUR, FIRST
SEASONS = range(FIRST, CUR + 1)
HALF_LIFE = 10  # games; carries across seasons, acts as regression to recent form


def load_games():
    g = pd.read_csv(f'{DATA}/games.csv')
    g = g[g.season.isin(SEASONS)].copy()
    g['gameday'] = pd.to_datetime(g.gameday)
    return g.sort_values(['gameday', 'game_id']).reset_index(drop=True)


def team_game_stats():
    """One row per team per game: offensive EPA/play split pass/rush, success, plays, points."""
    cols = ['game_id', 'season', 'week', 'posteam', 'defteam', 'play_type', 'epa', 'success',
            'pass', 'rush', 'qb_dropback', 'game_date', 'home_team', 'away_team', 'total_home_score',
            'total_away_score', 'wp', 'down', 'yards_gained', 'cpoe', 'pass_oe', 'sack']
    rows = []
    for y in SEASONS:
        p = pd.read_parquet(f'{DATA}/pbp_{y}.parquet', columns=cols)
        p = p[p.play_type.isin(['pass', 'run']) & p.epa.notna() & p.posteam.notna()]
        # garbage time filter: keep plays with win prob 5%-95% for rating purposes
        p['live'] = p.wp.between(0.05, 0.95)
        grp = p.groupby(['game_id', 'posteam', 'defteam'])
        t = pd.DataFrame({
            'plays': grp.size(),
            'epa': grp.epa.mean(),
            'epa_live': grp.apply(lambda d: d.loc[d.live, 'epa'].mean(), include_groups=False),
            'pass_epa': grp.apply(lambda d: d.loc[d['pass'] == 1, 'epa'].mean(), include_groups=False),
            'rush_epa': grp.apply(lambda d: d.loc[d['rush'] == 1, 'epa'].mean(), include_groups=False),
            'success': grp.success.mean(),
            'pass_rate': grp['pass'].mean(),
            'early_epa': grp.apply(lambda d: d.loc[d.down <= 2, 'epa'].mean(), include_groups=False),
            'explosive': grp.apply(lambda d: ((d['pass'] == 1) & (d.yards_gained >= 20) |
                                              (d['rush'] == 1) & (d.yards_gained >= 10)).mean(), include_groups=False),
            'cpoe': grp.cpoe.mean(),
            'proe': grp.pass_oe.mean(),
            'sack_rate': grp.apply(lambda d: d.loc[d.qb_dropback == 1, 'sack'].mean(), include_groups=False),
        }).reset_index()
        # PFR: share of dropbacks pressured (offense = allowed)
        pf = pd.read_parquet(f'{DATA}/pfr_pass_{y}.parquet')
        pf = pf.groupby(['game_id', 'team']).times_pressured_pct.mean().rename('pressure').reset_index()
        t = t.merge(pf.rename(columns={'team': 'posteam'}), on=['game_id', 'posteam'], how='left')
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def build():
    g = load_games()
    tg = team_game_stats().rename(columns={'posteam': 'team', 'defteam': 'opp'})
    # points from schedule
    long = pd.concat([
        g.assign(team=g.home_team, opp=g.away_team, pf=g.home_score, pa=g.away_score),
        g.assign(team=g.away_team, opp=g.home_team, pf=g.away_score, pa=g.home_score),
    ])[['game_id', 'gameday', 'season', 'week', 'team', 'opp', 'pf', 'pa']]
    long = long.merge(tg.drop(columns='opp'), on=['game_id', 'team'], how='left')
    # defensive = opponent's offensive numbers in same game
    d = tg.rename(columns={'team': 'opp', 'opp': 'team'})
    d = d[['game_id', 'team', 'plays', 'epa', 'epa_live', 'pass_epa', 'rush_epa', 'success', 'early_epa',
           'explosive', 'cpoe', 'sack_rate', 'pressure']]
    d.columns = ['game_id', 'team'] + ['d_' + c for c in d.columns[2:]]
    long = long.merge(d, on=['game_id', 'team'], how='left').sort_values(['team', 'gameday'])

    stat_cols = ['pf', 'pa', 'plays', 'epa_live', 'pass_epa', 'rush_epa', 'success', 'pass_rate',
                 'd_epa_live', 'd_pass_epa', 'd_rush_epa', 'd_success', 'd_plays', 'early_epa', 'explosive',
                 'cpoe', 'proe', 'sack_rate', 'pressure', 'd_early_epa', 'd_explosive', 'd_cpoe', 'd_sack_rate',
                 'd_pressure']
    played = long.pf.notna()
    out = []
    for team, df in long.groupby('team'):
        df = df.copy()
        hist = df[stat_cols].where(played.loc[df.index])
        # ewm of PRIOR games only -> shift(1) after ffill of the ewm over played games
        ew = hist.ewm(halflife=HALF_LIFE, ignore_na=True).mean().shift(1)
        ew = ew.ffill()
        n_prior = played.loc[df.index].astype(int).cumsum().shift(1).fillna(0)
        ew.columns = ['r_' + c for c in stat_cols]
        df = pd.concat([df[['game_id', 'team']], ew], axis=1)
        df['n_prior'] = n_prior.values
        out.append(df)
    ratings = pd.concat(out)

    h = ratings.add_prefix('h_').rename(columns={'h_game_id': 'game_id', 'h_team': 'home_team'})
    a = ratings.add_prefix('a_').rename(columns={'a_game_id': 'game_id', 'a_team': 'away_team'})
    X = g.merge(h, on=['game_id', 'home_team']).merge(a, on=['game_id', 'away_team'])
    X['neutral'] = (X.location == 'Neutral').astype(int)
    X['rest_diff'] = (X.home_rest - X.away_rest).clip(-7, 7)
    X['dome'] = X.roof.isin(['dome', 'closed']).astype(int)
    X['wind_f'] = X.wind.fillna(0).where(X.dome == 0, 0)
    # matchup features
    X['m_home_off_vs_def'] = X.h_r_epa_live + X.a_r_d_epa_live   # home O EPA + EPA away D allows
    X['m_away_off_vs_def'] = X.a_r_epa_live + X.h_r_d_epa_live
    X['m_net'] = X.m_home_off_vs_def - X.m_away_off_vs_def
    X['m_pass'] = (X.h_r_pass_epa + X.a_r_d_pass_epa) - (X.a_r_pass_epa + X.h_r_d_pass_epa)
    X['m_rush'] = (X.h_r_rush_epa + X.a_r_d_rush_epa) - (X.a_r_rush_epa + X.h_r_d_rush_epa)
    X['m_pts'] = (X.h_r_pf - X.h_r_pa) - (X.a_r_pf - X.a_r_pa)
    X['t_pts'] = X.h_r_pf + X.h_r_pa + X.a_r_pf + X.a_r_pa
    X['t_epa'] = X.m_home_off_vs_def + X.m_away_off_vs_def
    # advanced matchups (home minus away; defense 'd_' = what it allows)
    X['m_early'] = (X.h_r_early_epa + X.a_r_d_early_epa) - (X.a_r_early_epa + X.h_r_d_early_epa)
    X['m_expl'] = (X.h_r_explosive + X.a_r_d_explosive) - (X.a_r_explosive + X.h_r_d_explosive)
    X['m_press'] = (X.h_r_pressure + X.a_r_d_pressure) - (X.a_r_pressure + X.h_r_d_pressure)  # >0 = home QB under more pressure
    X['m_cpoe'] = (X.h_r_cpoe + X.a_r_d_cpoe) - (X.a_r_cpoe + X.h_r_d_cpoe)
    X['t_expl'] = X.h_r_explosive + X.a_r_d_explosive + X.a_r_explosive + X.h_r_d_explosive
    X['t_proe'] = X.h_r_proe + X.a_r_proe
    X['t_press'] = X.h_r_pressure + X.a_r_d_pressure + X.a_r_pressure + X.h_r_d_pressure
    X['t_plays'] = X.h_r_plays + X.h_r_d_plays + X.a_r_plays + X.a_r_d_plays
    return X


if __name__ == '__main__':
    X = build()
    X.to_pickle(f'{DATA}/features.pkl')
    print(X.shape, X[X.season == CUR][['week', 'home_team', 'away_team', 'm_net', 'm_pts', 'n_prior' if 'n_prior' in X else 'h_n_prior']].tail(5))
