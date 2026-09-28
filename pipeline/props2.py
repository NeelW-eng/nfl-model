"""Usage x efficiency prop models built on nflverse + Next Gen Stats + PFR advanced + snap counts + pbp.

Every feature for a player-game uses only games BEFORE it (EWMA then shift(1)).
Models: gradient-boosted trees per stat, trained walk-forward.
"""
import pandas as pd, numpy as np, json
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from paths import DATA, CUR, FIRST
YEARS = range(FIRST, CUR + 1)
TARGETS = {  # stat: (positions, loss)
    'passing_yards': (['QB'], 'squared_error'), 'passing_tds': (['QB'], 'poisson'),
    'rushing_yards': (['QB', 'RB', 'WR'], 'squared_error'), 'receiving_yards': (['WR', 'TE', 'RB'], 'squared_error'),
    'receptions': (['WR', 'TE', 'RB'], 'poisson'), 'anytime_td': (['QB', 'RB', 'WR', 'TE'], 'class'),
}


def ewm_prior(df, keys, cols, hl, sort='t', prefix='e_'):
    df = df.sort_values(keys + [sort])
    g = df.groupby(keys)[cols]
    out = g.transform(lambda s: s.ewm(halflife=hl, ignore_na=True).mean().shift(1))
    out.columns = [prefix + c for c in cols]
    return pd.concat([df, out], axis=1)


# ------------------------------------------------------------------ raw tables
def base_players():
    ps = pd.concat([pd.read_parquet(f'{DATA}/ps_{y}.parquet') for y in YEARS])
    ps = ps[(ps.season_type == 'REG') & ps.position.isin(['QB', 'RB', 'WR', 'TE'])].copy()
    ps['t'] = ps.season * 100 + ps.week
    ps['tds'] = ps.rushing_tds.fillna(0) + ps.receiving_tds.fillna(0)
    ps['anytime_td'] = (ps.tds > 0).astype(float)
    return ps


def pbp_player_and_team():
    cols = ['game_id', 'season', 'week', 'posteam', 'defteam', 'play_type', 'epa', 'pass_oe', 'yardline_100',
            'receiver_player_id', 'rusher_player_id', 'air_yards', 'pass_attempt', 'rush_attempt', 'wp',
            'home_team', 'qb_dropback', 'sack', 'success']
    P, T, D = [], [], []
    for y in YEARS:
        p = pd.read_parquet(f'{DATA}/pbp_{y}.parquet', columns=cols)
        p = p[p.play_type.isin(['pass', 'run'])]
        p['t'] = y * 100 + p.week
        rz = p.yardline_100 <= 20
        # player opportunity detail
        rec = p[p.receiver_player_id.notna()].assign(
            rz=rz, deep=p.air_yards >= 20).groupby(['receiver_player_id', 't']).agg(
            rz_targets=('rz', 'sum'), deep_targets=('deep', 'sum'), air_yards_tot=('air_yards', 'sum'))
        rec.index.names = ['player_id', 't']
        ru = p[p.rusher_player_id.notna()].assign(rz=rz, gl=p.yardline_100 <= 5).groupby(
            ['rusher_player_id', 't']).agg(rz_carries=('rz', 'sum'), gl_carries=('gl', 'sum'))
        ru.index.names = ['player_id', 't']
        P.append(rec.join(ru, how='outer'))
        # team offense per game
        neutral = p.wp.between(0.2, 0.8)
        tm = p.groupby(['posteam', 't']).agg(
            team_plays=('epa', 'size'), team_pass_att=('pass_attempt', 'sum'), team_rush_att=('rush_attempt', 'sum'),
            team_proe=('pass_oe', 'mean'), team_rz_plays=('yardline_100', lambda s: (s <= 20).sum()),
            team_epa=('epa', 'mean'))
        tm['team_neutral_proe'] = p[neutral].groupby(['posteam', 't']).pass_oe.mean()
        tm.index.names = ['team', 't']
        T.append(tm)
        # defense per game (what it allows)
        pa = p[p.pass_attempt == 1].groupby(['defteam', 't']).epa.mean().rename('d_pass_epa')
        ra = p[p.rush_attempt == 1].groupby(['defteam', 't']).epa.mean().rename('d_rush_epa')
        sk = p[p.qb_dropback == 1].groupby(['defteam', 't']).sack.mean().rename('d_sack_rate')
        dd = pd.concat([pa, ra, sk], axis=1); dd.index.names = ['opp', 't']
        D.append(dd)
    return pd.concat(P).reset_index(), pd.concat(T).reset_index(), pd.concat(D).reset_index()


def ngs():
    out = []
    spec = {'receiving': ['avg_separation', 'avg_cushion', 'avg_yac_above_expectation', 'avg_intended_air_yards',
                          'percent_share_of_intended_air_yards', 'catch_percentage'],
            'rushing': ['efficiency', 'percent_attempts_gte_eight_defenders', 'rush_yards_over_expected_per_att',
                        'avg_time_to_los'],
            'passing': ['avg_time_to_throw', 'completion_percentage_above_expectation', 'aggressiveness',
                        'avg_intended_air_yards', 'avg_air_yards_to_sticks']}
    for k, cols in spec.items():
        n = pd.read_parquet(f'{DATA}/ngs_{k}.parquet')
        n = n[(n.week > 0) & (n.season_type == 'REG')]
        n = n[['player_gsis_id', 'season', 'week'] + cols].rename(columns={'player_gsis_id': 'player_id'})
        n.columns = ['player_id', 'season', 'week'] + [f'ngs_{k[:3]}_{c}' for c in cols]
        out.append(n.groupby(['player_id', 'season', 'week']).mean().reset_index())
    m = out[0]
    for o in out[1:]:
        m = m.merge(o, on=['player_id', 'season', 'week'], how='outer')
    return m


def pfr_and_snaps():
    ids = pd.read_parquet(f'{DATA}/players.parquet', columns=['gsis_id', 'pfr_id']).dropna().drop_duplicates('pfr_id')
    ids = dict(zip(ids.pfr_id, ids.gsis_id))
    frames = []
    for y in YEARS:
        s = pd.read_parquet(f'{DATA}/snaps_{y}.parquet')
        s = s[s.game_type == 'REG'][['pfr_player_id', 'season', 'week', 'offense_pct']]
        pa = pd.read_parquet(f'{DATA}/pfr_pass_{y}.parquet')
        pa = pa[pa.game_type == 'REG'][['pfr_player_id', 'season', 'week', 'times_pressured_pct', 'passing_bad_throw_pct']]
        ru = pd.read_parquet(f'{DATA}/pfr_rush_{y}.parquet')
        ru = ru[ru.game_type == 'REG'][['pfr_player_id', 'season', 'week', 'rushing_yards_after_contact_avg',
                                       'rushing_broken_tackles']]
        rc = pd.read_parquet(f'{DATA}/pfr_rec_{y}.parquet')
        rc = rc[rc.game_type == 'REG'][['pfr_player_id', 'season', 'week', 'receiving_drop_pct',
                                       'receiving_broken_tackles']]
        f = s
        for o in (pa, ru, rc):
            f = f.merge(o, on=['pfr_player_id', 'season', 'week'], how='outer')
        frames.append(f)
    f = pd.concat(frames)
    f['player_id'] = f.pfr_player_id.map(ids)
    f = f.dropna(subset=['player_id']).drop(columns='pfr_player_id')
    f.columns = [c if c in ('player_id', 'season', 'week', 'offense_pct') else 'pfr_' + c for c in f.columns]
    return f.groupby(['player_id', 'season', 'week']).mean().reset_index()


def pressure_by_defense():
    """share of opponent dropbacks pressured, from PFR QB-level data aggregated to the defense."""
    fr = []
    for y in YEARS:
        pa = pd.read_parquet(f'{DATA}/pfr_pass_{y}.parquet')
        pa = pa[pa.game_type == 'REG']
        fr.append(pa.groupby(['opponent', 'season', 'week']).times_pressured_pct.mean().reset_index())
    f = pd.concat(fr).rename(columns={'opponent': 'opp', 'times_pressured_pct': 'd_pressure_pct'})
    f['t'] = f.season * 100 + f.week
    return f[['opp', 't', 'd_pressure_pct']]


# ------------------------------------------------------------------ feature table
USAGE = ['offense_pct', 'targets', 'target_share', 'air_yards_share', 'wopr', 'carries', 'carry_share', 'attempts',
         'rz_targets', 'rz_carries', 'gl_carries', 'deep_targets', 'receptions', 'receiving_yards',
         'rushing_yards', 'passing_yards', 'passing_tds', 'tds', 'rz_share']
EFF = ['yprr_proxy', 'ypc', 'ypt', 'ngs_rec_avg_separation', 'ngs_rec_avg_cushion', 'ngs_rec_avg_yac_above_expectation',
       'ngs_rec_avg_intended_air_yards', 'ngs_rec_catch_percentage', 'ngs_rus_efficiency',
       'ngs_rus_percent_attempts_gte_eight_defenders', 'ngs_rus_rush_yards_over_expected_per_att',
       'ngs_pas_avg_time_to_throw', 'ngs_pas_completion_percentage_above_expectation', 'ngs_pas_aggressiveness',
       'ngs_pas_avg_intended_air_yards', 'pfr_times_pressured_pct', 'pfr_passing_bad_throw_pct',
       'pfr_rushing_yards_after_contact_avg', 'pfr_rushing_broken_tackles', 'pfr_receiving_drop_pct',
       'passing_epa', 'passing_cpoe', 'rushing_epa', 'receiving_epa', 'racr']
TEAM = ['team_plays', 'team_pass_att', 'team_rush_att', 'team_proe', 'team_neutral_proe', 'team_rz_plays', 'team_epa']
DEF = ['d_pass_epa', 'd_rush_epa', 'd_sack_rate', 'd_pressure_pct']


def build(future_week=None, teams=None):
    ps = base_players()
    if future_week:  # add rows for the upcoming games (stats unknown)
        g = pd.read_csv(f'{DATA}/games.csv')
        sch = g[(g.season == CUR) & (g.week == future_week)]
        opp = {**dict(zip(sch.home_team, sch.away_team)), **dict(zip(sch.away_team, sch.home_team))}
        cur = ps[ps.season == CUR].sort_values('t').groupby('player_id').tail(1)
        played = set(ps[(ps.season == CUR) & (ps.week == future_week)].team)
        cur = cur[cur.team.isin(opp) & ~cur.team.isin(played) & (cur.team.isin(teams) if teams else True)].copy()
        stat_cols = [c for c in ps.columns if c not in ('player_id', 'player_name', 'player_display_name', 'position',
                                                         'position_group', 'headshot_url', 'team')]
        cur[stat_cols] = np.nan
        cur['season'], cur['week'], cur['t'] = CUR, future_week, CUR * 100 + future_week
        cur['opponent_team'] = cur.team.map(opp)
        cur['season_type'] = 'REG'
        cur['future'] = True
        ps = pd.concat([ps, cur])
    ps['future'] = ps.get('future', False)
    ps['future'] = ps.future.fillna(False).astype(bool)
    pl, tm, df = pbp_player_and_team()
    ps = ps.merge(pl, on=['player_id', 't'], how='left')
    ps = ps.merge(ngs(), on=['player_id', 'season', 'week'], how='left')
    ps = ps.merge(pfr_and_snaps(), on=['player_id', 'season', 'week'], how='left')
    ps = ps.merge(tm, on=['team', 't'], how='left')
    ps['carry_share'] = ps.carries / ps.team_rush_att
    ps['rz_share'] = (ps.rz_targets.fillna(0) + ps.rz_carries.fillna(0)) / ps.team_rz_plays.replace(0, np.nan)
    ps['ypc'] = ps.rushing_yards / ps.carries.replace(0, np.nan)
    ps['ypt'] = ps.receiving_yards / ps.targets.replace(0, np.nan)
    ps['yprr_proxy'] = ps.receiving_yards / (ps.offense_pct * ps.team_pass_att).replace(0, np.nan)
    played = ~ps.future
    for c in ['rz_targets', 'rz_carries', 'gl_carries', 'deep_targets']:
        ps.loc[played, c] = ps.loc[played, c].fillna(0)
    # player history
    ps = ewm_prior(ps, ['player_id'], USAGE, hl=4, prefix='u_')
    ps = ewm_prior(ps, ['player_id'], EFF, hl=8, prefix='f_')
    ps['last_offense_pct'] = ps.groupby('player_id').offense_pct.shift(1)
    ps['last_target_share'] = ps.groupby('player_id').target_share.shift(1)
    ps['last_carry_share'] = ps.groupby('player_id').carry_share.shift(1)
    ps['n_prior'] = ps.groupby('player_id').cumcount()
    ps['team_changed'] = (ps.groupby('player_id').team.shift(1) != ps.team).astype(int)
    # team environment (prior games of this team)
    tt = tm.sort_values('t')
    tt = ewm_prior(tt, ['team'], TEAM, hl=6, prefix='tm_')[['team', 't'] + ['tm_' + c for c in TEAM]]
    if future_week:
        last = tt.sort_values('t').groupby('team').tail(1).copy()
        full = tm.sort_values('t').groupby('team')[TEAM].apply(lambda d: d.ewm(halflife=6).mean().iloc[-1])
        last = full.add_prefix('tm_').reset_index(); last['t'] = CUR * 100 + future_week
        tt = pd.concat([tt, last]).drop_duplicates(['team', 't'], keep='first')  # real rows win
    ps = ps.merge(tt, on=['team', 't'], how='left')
    # defense faced
    dd = df.merge(pressure_by_defense(), on=['opp', 't'], how='outer').sort_values('t')
    ddp = ewm_prior(dd, ['opp'], DEF, hl=8, prefix='df_')[['opp', 't'] + ['df_' + c for c in DEF]]
    if future_week:
        full = dd.sort_values('t').groupby('opp')[DEF].apply(lambda d: d.ewm(halflife=8).mean().iloc[-1])
        last = full.add_prefix('df_').reset_index(); last['t'] = CUR * 100 + future_week
        ddp = pd.concat([ddp, last]).drop_duplicates(['opp', 't'], keep='first')
    ps = ps.merge(ddp.rename(columns={'opp': 'opponent_team'}), on=['opponent_team', 't'], how='left')
    # position-level yards allowed by this defense (prior)
    for stat in ['receiving_yards', 'rushing_yards', 'passing_yards', 'receptions', 'tds']:
        a = ps[~ps.future].groupby(['opponent_team', 'position', 't'])[stat].sum().reset_index()
        lg = a.groupby(['position', 't'])[stat].transform('mean')
        a['r'] = a[stat] / lg.replace(0, np.nan)
        a = a.sort_values('t')
        a['df_allow_' + stat] = a.groupby(['opponent_team', 'position']).r.transform(
            lambda s: s.ewm(halflife=8).mean().shift(1))
        if future_week:
            lastv = a.groupby(['opponent_team', 'position']).r.apply(lambda s: s.ewm(halflife=8).mean().iloc[-1])
            lastv = lastv.rename('df_allow_' + stat).reset_index(); lastv['t'] = CUR * 100 + future_week
            a = pd.concat([a, lastv]).drop_duplicates(['opponent_team', 'position', 't'], keep='first')
        ps = ps.merge(a[['opponent_team', 'position', 't', 'df_allow_' + stat]],
                      on=['opponent_team', 'position', 't'], how='left')
    # game environment from the market line + weather
    g = pd.read_csv(f'{DATA}/games.csv')
    g = g[g.season.isin(YEARS)]
    env = pd.concat([
        pd.DataFrame({'game_id': g.game_id, 'team': g.home_team, 'is_home': 1,
                      'implied_pts': g.total_line / 2 + g.spread_line / 2, 'game_total': g.total_line,
                      'spread_for': g.spread_line, 'dome': g.roof.isin(['dome', 'closed']).astype(int),
                      'wind': g.wind.fillna(0)}),
        pd.DataFrame({'game_id': g.game_id, 'team': g.away_team, 'is_home': 0,
                      'implied_pts': g.total_line / 2 - g.spread_line / 2, 'game_total': g.total_line,
                      'spread_for': -g.spread_line, 'dome': g.roof.isin(['dome', 'closed']).astype(int),
                      'wind': g.wind.fillna(0)})])
    env['season'] = env.game_id.str[:4].astype(int); env['week'] = env.game_id.str[5:7].astype(int)
    ps = ps.drop(columns=[c for c in ['game_id'] if c in ps]).merge(
        env.drop(columns='game_id'), on=['team', 'season', 'week'], how='left')
    ps['pos_code'] = ps.position.map({'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3})
    return ps


def feature_cols(ps):
    return (['u_' + c for c in USAGE] + ['f_' + c for c in EFF] +
            ['last_offense_pct', 'last_target_share', 'last_carry_share', 'n_prior', 'team_changed', 'pos_code'] +
            ['tm_' + c for c in TEAM] + ['df_' + c for c in DEF] +
            [c for c in ps.columns if c.startswith('df_allow_')] +
            ['is_home', 'implied_pts', 'game_total', 'spread_for', 'dome', 'wind'])


def train_rows(ps, stat):
    pos = TARGETS[stat][0]
    d = ps[ps.position.isin(pos) & (ps.n_prior >= 1) & (ps.season >= 2021)]
    # only players with a real role: prior snaps or prior touches
    role = (d.u_offense_pct.fillna(0) > 0.2) | (d.u_targets.fillna(0) > 1) | (d.u_carries.fillna(0) > 2) | \
           (d.u_attempts.fillna(0) > 10)
    return d[role]


def make_model(stat):
    loss = TARGETS[stat][1]
    kw = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=1.0,
              early_stopping=False, random_state=0)
    if loss == 'class':
        return HistGradientBoostingClassifier(**kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


def target(d, stat):
    return d[stat].fillna(0) if stat != 'anytime_td' else d.anytime_td


def fit_predict(train, test, stat, cols):
    m = make_model(stat)
    m.fit(train[cols], target(train, stat))
    if stat == 'anytime_td':
        return m, m.predict_proba(test[cols])[:, 1]
    return m, m.predict(test[cols])


def backtest(ps, old=None):
    cols = feature_cols(ps)
    res, resid = {}, {}
    for stat in TARGETS:
        d = train_rows(ps[~ps.future], stat)
        preds = []
        for s in [2024, 2025]:
            tr, te = d[d.season < s], d[d.season == s].copy()
            _, te['pred'] = fit_predict(tr, te, stat, cols)
            preds.append(te)
        te = pd.concat(preds)
        y = target(te, stat)
        r = dict(n=len(te))
        if stat == 'anytime_td':
            r['brier_new'] = ((te.pred - y) ** 2).mean()
            base = y.groupby(te.position).transform('mean')
            r['brier_base'] = ((base - y) ** 2).mean()
        else:
            r['mae_new'] = (te.pred - y).abs().mean()
        if old is not None:
            o = te[['player_id', 't']].merge(old[stat], on=['player_id', 't'], how='left')
            ok = o.proj.notna().values
            if stat == 'anytime_td':
                r['brier_old'] = ((1 - np.exp(-o.proj[ok].values) - y.values[ok]) ** 2).mean()
                r['brier_new_same'] = ((te.pred.values[ok] - y.values[ok]) ** 2).mean()
            else:
                r['mae_old_same'] = np.abs(o.proj[ok].values - y.values[ok]).mean()
                r['mae_new_same'] = np.abs(te.pred.values[ok] - y.values[ok]).mean()
        res[stat] = r
        resid[stat] = te[['player_id', 't', 'position', 'pred']].assign(y=y.values)
    return pd.DataFrame(res).T, resid


# ------------------------------------------------------------------ production engine
YARDS = ['passing_yards', 'rushing_yards', 'receiving_yards']
BLEND = {'passing_tds': 0.7, 'receptions': 0.6, 'anytime_td': 0.7}  # weight on the boosted model vs simple EWMA


def quantile_model(q):
    return HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=400, learning_rate=0.04,
                                         max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)


def split_normal_over(line, q25, q50, q75):
    """P(X > line) from three quantiles (split normal around the median)."""
    from scipy.stats import norm
    lo = max((q50 - q25) / 0.6745, 3.0)
    hi = max((q75 - q50) / 0.6745, 3.0)
    return float(1 - norm.cdf((line - q50) / hi)) if line >= q50 else float(norm.cdf((q50 - line) / lo))


def fit_quantiles(train, test, stat, cols):
    out = {}
    for q in (0.25, 0.5, 0.75):
        m = quantile_model(q).fit(train[cols], train[stat].fillna(0))
        out[q] = m.predict(test[cols])
    q25, q50, q75 = np.minimum(out[0.25], out[0.5]), out[0.5], np.maximum(out[0.75], out[0.5])
    return q25, q50, q75


def prob_over(stat, r, line):
    """P(over line) for one projection row r (dict-like)."""
    from scipy.stats import norm, poisson, nbinom
    if stat in YARDS:
        import json
        c = json.load(open(f'{DATA}/quantile_cal.json'))[stat]
        lo = max((r['q50'] - r['q25']) / 0.6745, 3) * c['s_lo']
        hi = max((r['q75'] - r['q50']) / 0.6745, 3) * c['s_hi']
        m = r['q50'] + (c['shift'] * (hi / c['s_hi']) if c['shift'] >= 0 else c['shift'] * (lo / c['s_lo']))
        p = 1 - norm.cdf((line - m) / hi) if line >= m else norm.cdf((m - line) / lo)
        return float(0.5 + c['shrink'] * (p - 0.5))
    mu = max(r['proj'], 0.01)
    if stat == 'receptions':
        k = 12.0
        return float(1 - nbinom.cdf(np.floor(line), k, k / (k + mu)))
    if stat == 'passing_tds':
        return float(1 - poisson.cdf(np.floor(line), mu))
    if stat == 'anytime_td':
        return float(r['p_td'])


SIGNAL_COLS = ['u_offense_pct', 'last_offense_pct', 'u_target_share', 'last_target_share', 'u_carry_share',
               'last_carry_share', 'u_rz_share', 'implied_pts', 'tm_team_proe', 'df_pass_epa', 'df_rush_epa',
               'df_pressure_pct', 'f_ngs_rec_avg_separation', 'f_ngs_rec_avg_yac_above_expectation',
               'f_ngs_rus_rush_yards_over_expected_per_att', 'f_ngs_pas_completion_percentage_above_expectation',
               'f_pfr_times_pressured_pct', 'f_pfr_rushing_yards_after_contact_avg', 'u_attempts', 'u_targets',
               'u_carries', 'n_prior']


def project_week(week, teams=None):
    """Train on all completed games, project the upcoming week. Returns {stat: DataFrame}."""
    import props as simple
    ps = build(future_week=week, teams=teams)
    cols = feature_cols(ps)
    fut = ps[ps.future].copy()
    import weekly as wk
    simple_proj = wk.upcoming_projections(week)[0]
    out = {}
    # league/position reference values for signals
    ref = ps[(~ps.future) & (ps.season >= 2024)]
    posref = {c: ref.groupby('position')[c].median() for c in SIGNAL_COLS if c.startswith('f_')}
    for stat in TARGETS:
        pos = TARGETS[stat][0]
        tr = train_rows(ps[~ps.future], stat)
        te = fut[fut.position.isin(pos)].copy()
        te = te[(te.u_offense_pct.fillna(0) > 0.25) | (te.u_targets.fillna(0) > 2) | (te.u_carries.fillna(0) > 4) |
                (te.u_attempts.fillna(0) > 15)]
        if te.empty:
            continue
        if stat in YARDS:
            m = quantile_model(0.5).fit(tr[cols], tr[stat].fillna(0))
            te['q50'] = np.maximum(m.predict(te[cols]), 0.5)
            ratios = json.load(open(f'{DATA}/quantile_cal.json'))[stat]['ratios']
            dflt = list(ratios.values())[-1]
            te['q25'] = te.q50 * te.position.map(lambda p: ratios.get(p, dflt)[0])
            te['q75'] = te.q50 * te.position.map(lambda p: ratios.get(p, dflt)[1])
            te['proj'] = te.q50
        else:
            _, pred = fit_predict(tr, te, stat, cols)
            # simple EWMA model for the blend
            s = simple_proj[stat].drop_duplicates('player_id').set_index('player_id').proj
            old = te.player_id.map(s)
            w = BLEND[stat]
            if stat == 'anytime_td':
                pred_s = pd.Series(pred, index=te.index)
                p_old = (1 - np.exp(-old)).fillna(pred_s)
                te['p_td'] = w * pred_s + (1 - w) * p_old
                te['proj'] = -np.log(1 - te.p_td.clip(1e-3, .999))  # expected TDs equivalent
            else:
                te['proj'] = w * pred + (1 - w) * old.fillna(pd.Series(pred, index=te.index))
        for c in posref:
            te['ref_' + c] = te.position.map(posref[c])
        out[stat] = te
    return out
