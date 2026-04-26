"""
NFL 2-Point Conversion Analysis v9 — Direct WP Model + EPA Analysis
=====================================================================
Two clean layers:
1. WP Grid: Pre-computed from nflfastR's actual XGBoost model via R script.
   Inputs: margin (7/8/9), time, home/away, timeouts for each team.
   No spread, no EPA — pure game-state WP for an average team.

2. EPA Analysis: Real game data filtered by team quality. Scenario
   explorer, decision lab, comeback rates — all using actual PBP.

Run order:
  1. Rscript generate_wp_grid.R   (creates wp_grid.csv — run once)
  2. python process_data.py        (creates dashboard_data.json)
  3. Open dashboard.html

Install:  pip install nfl_data_py pandas numpy
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
from pathlib import Path


def load_pbp(seasons=range(2006, 2026)):
    for name, fn in [
        ("nfl_data_py", lambda: __import__('nfl_data_py').import_pbp_data(list(seasons))),
        ("nflreadpy",   lambda: __import__('nflreadpy').load_pbp(list(seasons)).to_pandas()),
    ]:
        try:
            print(f"Trying {name}..."); df = fn()
            print(f"  Loaded {len(df):,} plays via {name}"); return df
        except Exception as e:
            print(f"  {name} failed: {e}")
    print("Trying direct parquet URLs...")
    frames = []
    for yr in seasons:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.parquet"
        try: frames.append(pd.read_parquet(url)); print(f"  {yr} OK")
        except: print(f"  {yr} skipped")
    if frames: return pd.concat(frames, ignore_index=True)
    raise RuntimeError("Could not load data.")


def build_team_epa(pbp):
    plays = pbp[
        pbp['play_type'].isin(['run', 'pass']) &
        pbp['epa'].notna() & pbp['posteam'].notna() &
        (pbp['season_type'] == 'REG')
    ]
    off = (plays.groupby(['season', 'posteam'])['epa']
           .mean().reset_index().rename(columns={'posteam': 'team', 'epa': 'off_epa'}))
    defense = (plays.groupby(['season', 'defteam'])['epa']
               .mean().reset_index().rename(columns={'defteam': 'team', 'epa': 'def_epa'}))
    r = off.merge(defense, on=['season', 'team'], how='outer')
    r['net_epa'] = r['off_epa'] - r['def_epa']
    return r


def compute_conversion_rates(pbp):
    xp = pbp[pbp['extra_point_result'].notna()].copy()
    xp['good'] = (xp['extra_point_result'] == 'good').astype(int)
    xp_yr = xp.groupby('season').agg(
        xp_attempts=('good', 'count'), xp_made=('good', 'sum')).reset_index()
    xp_yr['xp_rate'] = xp_yr['xp_made'] / xp_yr['xp_attempts']
    tp = pbp[pbp['two_point_conv_result'].notna()].copy()
    tp['good'] = (tp['two_point_conv_result'] == 'success').astype(int)
    tp_yr = tp.groupby('season').agg(
        twopt_attempts=('good', 'count'), twopt_made=('good', 'sum')).reset_index()
    tp_yr['twopt_rate'] = tp_yr['twopt_made'] / tp_yr['twopt_attempts']
    return xp_yr.merge(tp_yr, on='season', how='outer')


# ═══════════════════════════════════════════════════════════════════════
# LOAD WP GRID
# ═══════════════════════════════════════════════════════════════════════

def load_wp_grid():
    """Load the pre-computed WP grid from the R script output."""
    grid_path = Path('wp_grid.csv')
    if not grid_path.exists():
        raise FileNotFoundError(
            "wp_grid.csv not found. Run 'Rscript generate_wp_grid.R' first.")

    df = pd.read_csv(grid_path)
    # R writes TRUE/FALSE — convert to Python bool then to int for JSON
    df['trailing_is_home'] = df['trailing_is_home'].map(
        {True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 1: 1, 0: 0})
    print(f"\nLoaded WP grid: {len(df):,} rows")

    # Sanity check
    for h, label in [(1, "Trailing HOME"), (0, "Trailing AWAY")]:
        print(f"\n  {label}, 5 min, 3v3 TOs:")
        for m in [7, 8, 9]:
            sub = df[(df['margin'] == m) &
                     (df['game_seconds_remaining'] == 300) &
                     (df['trailing_timeouts'] == 3) &
                     (df['leading_timeouts'] == 3) &
                     (df['trailing_is_home'] == h)]
            if len(sub) > 0:
                print(f"    Down {m}: trailing WP = {sub['trailing_wp'].iloc[0]:.4f}, "
                      f"leading WP = {sub['leading_wp'].iloc[0]:.4f}")

    # Round for file size
    df['leading_wp'] = df['leading_wp'].round(5)
    df['trailing_wp'] = df['trailing_wp'].round(5)

    return df.to_dict('records')


# ═══════════════════════════════════════════════════════════════════════
# PAT DECISION LAB
# ═══════════════════════════════════════════════════════════════════════

def build_decision_lab(pbp, ratings):
    print("\nBuilding PAT decision lab (Q4 only)...")
    pat = pbp[
        ((pbp['play_type'] == 'extra_point') | (pbp['two_point_conv_result'].notna()))
    ].copy()
    pat = pat[pat['score_differential'] == 7].copy()

    # Q4 only
    pat = pat[pat['qtr'] == 4].copy()

    pat['chose_2pt'] = pat['two_point_conv_result'].notna().astype(int)
    pat['chose_xp'] = (pat['play_type'] == 'extra_point').astype(int)
    pat['xp_good'] = (pat['extra_point_result'] == 'good').astype(int)
    pat['twopt_good'] = (pat['two_point_conv_result'] == 'success').astype(int)
    pat['post_pat_margin'] = np.where(
        pat['chose_xp'] == 1,
        np.where(pat['xp_good'] == 1, 8, 7),
        np.where(pat['twopt_good'] == 1, 9, 7))
    pat['scoring_team'] = pat['posteam']
    pat['opponent'] = pat['defteam']
    pat['is_playoff'] = (pat['season_type'] == 'POST').astype(int)
    pat['scoring_is_home'] = pat['scoring_team'] == pat['home_team']
    # Tie counts as opponent came back (scoring team didn't hold)
    pat['scoring_team_won'] = np.where(pat['scoring_is_home'], pat['result'] > 0, pat['result'] < 0)
    pat['scoring_team_held'] = np.where(pat['scoring_is_home'], pat['result'] > 0, pat['result'] < 0)  # win only
    pat['opponent_came_back'] = np.where(
        pat['scoring_is_home'],
        pat['result'] <= 0,  # tie or loss for home = opponent came back
        pat['result'] >= 0   # tie or loss for away = opponent came back
    ).astype(int)
    pat['minutes_left'] = pat['game_seconds_remaining'].fillna(0) / 60.0
    pat['wp_at_plus7'] = pat['wp']

    # Get WP on next play
    pbp_s = pbp.sort_values(['game_id', 'play_id']).reset_index(drop=True)
    pbp_s['next_wp'] = pbp_s.groupby('game_id')['wp'].shift(-1)
    pbp_s['next_posteam'] = pbp_s.groupby('game_id')['posteam'].shift(-1)
    pat = pat.merge(pbp_s[['game_id', 'play_id', 'next_wp', 'next_posteam']],
                     on=['game_id', 'play_id'], how='left')
    pat['wp_after_pat'] = np.where(
        pat['next_posteam'] == pat['scoring_team'], pat['next_wp'], 1 - pat['next_wp'])

    # Home/away context
    pat['scoring_is_home_int'] = pat['scoring_is_home'].astype(int)

    # Timeouts at time of PAT
    pat['scoring_tos'] = pat['posteam_timeouts_remaining'].fillna(3)
    pat['opp_tos'] = pat['defteam_timeouts_remaining'].fillna(3)

    # Merge EPA
    pat = pat.merge(
        ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
            'team': 'scoring_team', 'off_epa': 'scoring_off_epa', 'def_epa': 'scoring_def_epa'}),
        on=['season', 'scoring_team'], how='left')
    pat = pat.merge(
        ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
            'team': 'opponent', 'off_epa': 'opp_off_epa', 'def_epa': 'opp_def_epa'}),
        on=['season', 'opponent'], how='left')

    keep = ['game_id', 'season', 'scoring_team', 'opponent',
            'chose_2pt', 'chose_xp', 'xp_good', 'twopt_good', 'post_pat_margin',
            'scoring_team_won', 'opponent_came_back', 'is_playoff', 'minutes_left',
            'wp_at_plus7', 'wp_after_pat',
            'scoring_is_home_int', 'scoring_tos', 'opp_tos',
            'scoring_off_epa', 'scoring_def_epa', 'opp_off_epa', 'opp_def_epa',
            'game_seconds_remaining', 'result']
    out = pat[[c for c in keep if c in pat.columns]].copy()
    for c in ['minutes_left', 'wp_at_plus7', 'wp_after_pat',
              'scoring_off_epa', 'scoring_def_epa', 'opp_off_epa', 'opp_def_epa']:
        if c in out.columns: out[c] = out[c].round(4)
    for c in ['chose_2pt', 'chose_xp', 'xp_good', 'twopt_good', 'post_pat_margin',
              'scoring_team_won', 'opponent_came_back', 'is_playoff', 'scoring_is_home_int']:
        if c in out.columns: out[c] = out[c].astype(int)
    for c in ['scoring_tos', 'opp_tos']:
        if c in out.columns: out[c] = out[c].round(0).astype(int)
    if 'result' in out.columns: out['result'] = out['result'].round(0).astype(int)
    if 'game_seconds_remaining' in out.columns:
        out['game_seconds_remaining'] = out['game_seconds_remaining'].round(0).astype(int)

    print(f"  {len(out)} PAT decisions")
    return out.to_dict('records')


# ═══════════════════════════════════════════════════════════════════════
# TRAILING SITUATIONS
# ═══════════════════════════════════════════════════════════════════════

def find_situations(pbp, ratings, max_seconds=600):
    q4 = pbp[
        (pbp['qtr'] == 4) & (pbp['game_seconds_remaining'] <= max_seconds) &
        pbp['game_seconds_remaining'].notna() & pbp['score_differential'].notna() &
        pbp['posteam'].notna()
    ].copy()
    rows = []
    for deficit in [7, 8, 9]:
        sub = q4[q4['score_differential'] == -deficit].copy()
        sub = sub.sort_values('game_seconds_remaining', ascending=False)
        first = sub.groupby(['game_id', 'posteam']).first().reset_index()
        first['deficit'] = deficit
        rows.append(first)
    df = pd.concat(rows, ignore_index=True)
    df['trailing_team'] = df['posteam']
    df['leading_team'] = df['defteam']
    df['won'] = np.where(df['posteam'] == df['home_team'], df['result'] > 0, df['result'] < 0)
    df['minutes_left'] = df['game_seconds_remaining'] / 60.0
    df['is_playoff'] = (df['season_type'] == 'POST').astype(int)
    df = df.merge(ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
        'team': 'trailing_team', 'off_epa': 'trail_off_epa', 'def_epa': 'trail_def_epa'}),
        on=['season', 'trailing_team'], how='left')
    df = df.merge(ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
        'team': 'leading_team', 'off_epa': 'lead_off_epa', 'def_epa': 'lead_def_epa'}),
        on=['season', 'leading_team'], how='left')
    keep = ['game_id', 'season', 'deficit', 'won', 'minutes_left', 'is_playoff',
            'trailing_team', 'leading_team',
            'trail_off_epa', 'trail_def_epa', 'lead_off_epa', 'lead_def_epa', 'result']
    out = df[[c for c in keep if c in df.columns]].copy()
    for c in ['minutes_left', 'trail_off_epa', 'trail_def_epa', 'lead_off_epa', 'lead_def_epa']:
        if c in out.columns: out[c] = out[c].round(4)
    out['won'] = out['won'].astype(int)
    if 'result' in out.columns: out['result'] = out['result'].round(0).astype(int)
    return out.to_dict('records')


def build_comeback_texture(pbp, ratings, max_seconds=600):
    """
    Find every Q4 drive start where the team with the ball is down 7, 8, or 9.
    Trigger: 1st & 10 plays where score_differential is -7/-8/-9 in Q4.

    Track the rest of the game:
    - peak_trailing_wp and initial_trailing_wp
    - thresholds: crossed 50%, crossed 30%, exceeded expected by 15pp, 30pp
    - went_to_ot, trailing_came_back, trailing_final_margin
    - EPA for both teams
    """
    print("\nBuilding comeback texture (Q4 drive starts, down 7/8/9)...")

    plays = pbp[
        pbp['wp'].notna() &
        pbp['game_seconds_remaining'].notna() &
        pbp['score_differential'].notna() &
        pbp['posteam'].notna()
    ].sort_values(['game_id', 'play_id']).copy()

    # Find first play of drives in Q4 where trailing team has ball
    drive_starts = plays[
        (plays['qtr'] == 4) &
        (plays['game_seconds_remaining'] <= max_seconds) &
        (plays['down'] == 1) &
        (plays['ydstogo'] == 10) &
        (plays['score_differential'].isin([-7, -8, -9]))
    ].copy()

    drive_starts['deficit'] = -drive_starts['score_differential']
    drive_starts['trailing_team'] = drive_starts['posteam']
    drive_starts['leading_team'] = drive_starts['defteam']

    # One per game per trailing team per deficit (take the first occurrence)
    drive_starts = drive_starts.sort_values('game_seconds_remaining', ascending=False)
    drive_starts = drive_starts.groupby(['game_id', 'trailing_team', 'deficit']).first().reset_index()

    print(f"  Found {len(drive_starts)} Q4 drive starts (down 7/8/9)")
    for d in [7, 8, 9]:
        print(f"    Down {d}: {len(drive_starts[drive_starts['deficit'] == d])}")

    results = []
    for _, row in drive_starts.iterrows():
        gid = row['game_id']
        trailing_team = row['trailing_team']
        trigger_play_id = row['play_id']
        trigger_seconds = row['game_seconds_remaining']
        deficit = int(row['deficit'])

        # Get all plays from trigger onward
        game_plays = plays[
            (plays['game_id'] == gid) &
            (plays['play_id'] >= trigger_play_id)
        ]
        if len(game_plays) == 0:
            continue

        # WP from trailing team's perspective
        game_plays = game_plays.copy()
        game_plays['trailing_wp'] = np.where(
            game_plays['posteam'] == trailing_team,
            game_plays['wp'],
            1 - game_plays['wp']
        )

        initial_wp = float(game_plays['trailing_wp'].iloc[0])
        peak_wp = float(game_plays['trailing_wp'].max())
        wp_above_expected = peak_wp - initial_wp
        went_to_ot = int((game_plays['qtr'] > 4).any())

        # Final result
        final_result = row['result']
        trailing_is_home = trailing_team == row['home_team']
        if trailing_is_home:
            trailing_came_back = int(final_result >= 0)
            trailing_won = int(final_result > 0)
            trailing_final_margin = int(final_result)
        else:
            trailing_came_back = int(final_result <= 0)
            trailing_won = int(final_result < 0)
            trailing_final_margin = int(-final_result)

        results.append({
            'game_id': gid,
            'season': int(row['season']),
            'trailing_team': trailing_team,
            'leading_team': row['leading_team'],
            'deficit': deficit,
            'minutes_left': round(trigger_seconds / 60, 1),
            'is_playoff': int(row['season_type'] == 'POST'),
            'initial_trailing_wp': round(initial_wp, 4),
            'peak_trailing_wp': round(peak_wp, 4),
            'wp_above_expected': round(wp_above_expected, 4),
            'crossed_50': int(peak_wp >= 0.50),
            'crossed_30': int(peak_wp >= 0.30),
            'exceeded_15': int(wp_above_expected >= 0.15),
            'exceeded_30': int(wp_above_expected >= 0.30),
            'went_to_ot': went_to_ot,
            'trailing_came_back': trailing_came_back,
            'trailing_won': trailing_won,
            'trailing_final_margin': trailing_final_margin,
        })

    df = pd.DataFrame(results)

    # Merge EPA
    df = df.merge(
        ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
            'team': 'trailing_team', 'off_epa': 'trail_off_epa', 'def_epa': 'trail_def_epa'}),
        on=['season', 'trailing_team'], how='left')
    df = df.merge(
        ratings[['season', 'team', 'off_epa', 'def_epa']].rename(columns={
            'team': 'leading_team', 'off_epa': 'lead_off_epa', 'def_epa': 'lead_def_epa'}),
        on=['season', 'leading_team'], how='left')

    for c in ['trail_off_epa', 'trail_def_epa', 'lead_off_epa', 'lead_def_epa']:
        if c in df.columns: df[c] = df[c].round(4)

    print(f"  {len(df)} situations tracked")
    for d in [7, 8, 9]:
        sub = df[df['deficit'] == d]
        if len(sub) == 0:
            continue
        expected = sub['initial_trailing_wp'].mean()
        actual = sub['trailing_came_back'].mean()
        print(f"\n  Down {d} ({len(sub)} drives):")
        print(f"    Peak WP median:      {sub['peak_trailing_wp'].median():.3f}")
        print(f"    Crossed 50% WP:      {sub['crossed_50'].sum()} ({100*sub['crossed_50'].mean():.1f}%)")
        print(f"    Crossed 30% WP:      {sub['crossed_30'].sum()} ({100*sub['crossed_30'].mean():.1f}%)")
        print(f"    Exceeded exp +15pp:  {sub['exceeded_15'].sum()} ({100*sub['exceeded_15'].mean():.1f}%)")
        print(f"    Exceeded exp +30pp:  {sub['exceeded_30'].sum()} ({100*sub['exceeded_30'].mean():.1f}%)")
        print(f"    Went to OT:          {sub['went_to_ot'].sum()} ({100*sub['went_to_ot'].mean():.1f}%)")
        print(f"    Came back (W+T):     {sub['trailing_came_back'].sum()} ({100*actual:.1f}%)")
        print(f"    Expected WP (avg):   {100*expected:.1f}%  →  Actual: {100*actual:.1f}%  (diff: {100*(actual-expected):+.1f} pp)")

    return df.to_dict('records')


def build_decision_texture(pbp, decisions_list, ratings):
    """
    For each PAT decision game, track the rest of the game from the
    OPPONENT's (trailing team's) perspective. Group by the actual
    post-PAT margin (+7, +8, or +9) to directly compare outcomes.
    """
    print("\nBuilding decision game texture...")

    plays = pbp[
        pbp['wp'].notna() & pbp['game_seconds_remaining'].notna() &
        pbp['score_differential'].notna() & pbp['posteam'].notna()
    ].sort_values(['game_id', 'play_id']).copy()

    decisions_df = pd.DataFrame(decisions_list)
    results = []

    for _, row in decisions_df.iterrows():
        gid = row['game_id']
        scoring_team = row['scoring_team']
        opponent = row['opponent']
        margin = int(row['post_pat_margin'])
        pat_seconds = row.get('game_seconds_remaining', 0)

        # Get all plays after the PAT
        game_plays = plays[
            (plays['game_id'] == gid) &
            (plays['game_seconds_remaining'] <= pat_seconds)
        ]
        if len(game_plays) == 0:
            continue

        # Track from the OPPONENT's (trailing team's) perspective
        game_plays = game_plays.copy()
        game_plays['trailing_wp'] = np.where(
            game_plays['posteam'] == opponent,
            game_plays['wp'],
            1 - game_plays['wp']
        )

        peak_wp = float(game_plays['trailing_wp'].max())
        went_to_ot = int((game_plays['qtr'] > 4).any())

        results.append({
            'game_id': gid,
            'season': int(row['season']),
            'scoring_team': scoring_team,
            'opponent': opponent,
            'chose_2pt': int(row['chose_2pt']),
            'post_pat_margin': margin,
            'scoring_team_won': int(row['scoring_team_won']),
            'opponent_came_back': int(row.get('opponent_came_back', 0)),
            'minutes_left': round(float(row.get('minutes_left', 0)), 1),
            'is_playoff': int(row.get('is_playoff', 0)),
            'peak_trailing_wp': round(peak_wp, 4),
            'crossed_50': int(peak_wp >= 0.50),
            'crossed_30': int(peak_wp >= 0.30),
            'went_to_ot': went_to_ot,
            'scoring_off_epa': row.get('scoring_off_epa'),
            'scoring_def_epa': row.get('scoring_def_epa'),
            'opp_off_epa': row.get('opp_off_epa'),
            'opp_def_epa': row.get('opp_def_epa'),
            'scoring_tos': row.get('scoring_tos'),
            'opp_tos': row.get('opp_tos'),
        })

    df = pd.DataFrame(results)
    print(f"  {len(df)} decision games tracked")
    for m in [7, 8, 9]:
        sub = df[df['post_pat_margin'] == m]
        if len(sub) == 0:
            continue
        print(f"\n  Margin +{m} ({len(sub)} games):")
        print(f"    Peak opp WP median: {sub['peak_trailing_wp'].median():.3f}")
        print(f"    Crossed 50%:        {sub['crossed_50'].sum()} ({100*sub['crossed_50'].mean():.1f}%)")
        print(f"    Crossed 30%:        {sub['crossed_30'].sum()} ({100*sub['crossed_30'].mean():.1f}%)")
        print(f"    Went to OT:         {sub['went_to_ot'].sum()}")
        print(f"    Scoring team won:   {sub['scoring_team_won'].sum()} ({100*sub['scoring_team_won'].mean():.1f}%)")

    return df.to_dict('records')


def build_epa_summary(texture_list):
    """
    Build EPA summaries from the Q4 kickoff texture data.
    Focuses on how team quality affects comeback rates at each deficit.
    """
    print("\nBuilding EPA summary from kickoff texture data...")
    df = pd.DataFrame(texture_list)
    if len(df) == 0:
        return {}

    summary = {}

    # EPA profiles by deficit: teams that came back vs held
    for d in [7, 8, 9]:
        sub = df[(df['deficit'] == d) &
                 df['trail_off_epa'].notna() & df['lead_off_epa'].notna()]
        came_back = sub[sub['trailing_came_back'] == 1]
        held = sub[sub['trailing_came_back'] == 0]

        summary[f'deficit_{d}'] = {
            'n': int(len(sub)),
            'comeback_rate': round(float(sub['trailing_came_back'].mean()), 4) if len(sub) > 0 else None,
            'scare_rate': round(float(sub['crossed_30'].mean()), 4) if len(sub) > 0 else None,
            'peak_wp_median': round(float(sub['peak_trailing_wp'].median()), 4) if len(sub) > 0 else None,
            'avg_minutes_left': round(float(sub['minutes_left'].mean()), 1) if len(sub) > 0 else None,
            'expected_wp': round(float(sub['initial_trailing_wp'].mean()), 4) if 'initial_trailing_wp' in sub.columns and len(sub) > 0 else None,
            'actual_comeback': round(float(sub['trailing_came_back'].mean()), 4) if len(sub) > 0 else None,
            'ot_rate': round(float(sub['went_to_ot'].mean()), 4) if len(sub) > 0 else None,
            # EPA when trailing team came back
            'cb_trail_off': round(float(came_back['trail_off_epa'].mean()), 4) if len(came_back) > 0 else None,
            'cb_trail_def': round(float(came_back['trail_def_epa'].mean()), 4) if len(came_back) > 0 else None,
            'cb_lead_off': round(float(came_back['lead_off_epa'].mean()), 4) if len(came_back) > 0 else None,
            'cb_lead_def': round(float(came_back['lead_def_epa'].mean()), 4) if len(came_back) > 0 else None,
            # EPA when leading team held
            'held_trail_off': round(float(held['trail_off_epa'].mean()), 4) if len(held) > 0 else None,
            'held_trail_def': round(float(held['trail_def_epa'].mean()), 4) if len(held) > 0 else None,
            'held_lead_off': round(float(held['lead_off_epa'].mean()), 4) if len(held) > 0 else None,
            'held_lead_def': round(float(held['lead_def_epa'].mean()), 4) if len(held) > 0 else None,
        }

    # Comeback/scare rates by EPA tier at each deficit
    valid = df[df['trail_off_epa'].notna() & df['lead_def_epa'].notna()]
    if len(valid) > 0:
        med_trail_off = valid['trail_off_epa'].median()
        med_lead_def = valid['lead_def_epa'].median()

        for tier_name, mask in [
            ('good_trail_off', valid['trail_off_epa'] >= med_trail_off),
            ('bad_trail_off', valid['trail_off_epa'] < med_trail_off),
            ('bad_lead_def', valid['lead_def_epa'] >= med_lead_def),
            ('good_lead_def', valid['lead_def_epa'] < med_lead_def),
        ]:
            tier = {}
            for d in [7, 8, 9]:
                sub = valid[(valid['deficit'] == d) & mask]
                tier[d] = {
                    'n': int(len(sub)),
                    'comeback_rate': round(float(sub['trailing_came_back'].mean()), 4) if len(sub) > 0 else None,
                    'scare_rate': round(float(sub['crossed_30'].mean()), 4) if len(sub) > 0 else None,
                    'peak_wp_median': round(float(sub['peak_trailing_wp'].median()), 4) if len(sub) > 0 else None,
                }
            summary[tier_name] = tier

    # 4x4 cross-tier grid: (trailing off × trailing def) × (leading off × leading def)
    if len(valid) > 0:
        med_trail_def = valid['trail_def_epa'].median()
        med_lead_off  = valid['lead_off_epa'].median()

        # trailing team tiers: off quality × def quality
        # Note: good defense = LOWER def_epa (fewer EPA allowed)
        trail_tiers = [
            ('good_off_good_def', (valid['trail_off_epa'] >= med_trail_off) & (valid['trail_def_epa'] <= med_trail_def)),
            ('good_off_bad_def',  (valid['trail_off_epa'] >= med_trail_off) & (valid['trail_def_epa'] >  med_trail_def)),
            ('bad_off_good_def',  (valid['trail_off_epa'] <  med_trail_off) & (valid['trail_def_epa'] <= med_trail_def)),
            ('bad_off_bad_def',   (valid['trail_off_epa'] <  med_trail_off) & (valid['trail_def_epa'] >  med_trail_def)),
        ]
        lead_tiers = [
            ('good_off_good_def', (valid['lead_off_epa'] >= med_lead_off) & (valid['lead_def_epa'] <= med_lead_def)),
            ('good_off_bad_def',  (valid['lead_off_epa'] >= med_lead_off) & (valid['lead_def_epa'] >  med_lead_def)),
            ('bad_off_good_def',  (valid['lead_off_epa'] <  med_lead_off) & (valid['lead_def_epa'] <= med_lead_def)),
            ('bad_off_bad_def',   (valid['lead_off_epa'] <  med_lead_off) & (valid['lead_def_epa'] >  med_lead_def)),
        ]

        cross_grid = {}
        for trail_label, trail_mask in trail_tiers:
            for lead_label, lead_mask in lead_tiers:
                cell_key = f'trail_{trail_label}__lead_{lead_label}'
                cell = {}
                for d in [7, 8, 9]:
                    sub = valid[(valid['deficit'] == d) & trail_mask & lead_mask]
                    cell[d] = {
                        'n': int(len(sub)),
                        'comeback_rate': round(float(sub['trailing_came_back'].mean()), 4) if len(sub) > 0 else None,
                        'scare_rate': round(float(sub['crossed_30'].mean()), 4) if len(sub) > 0 else None,
                        'peak_wp_median': round(float(sub['peak_trailing_wp'].median()), 4) if len(sub) > 0 else None,
                    }
                cross_grid[cell_key] = cell
        summary['cross_grid'] = cross_grid

        print("\n  Cross-tier grid (4×4: trailing [off×def] × leading [off×def]):")
        for cell_key, cell in cross_grid.items():
            rates = ', '.join([f"{d}={cell[d]['comeback_rate']}" for d in [7, 8, 9]])
            ns    = ', '.join([f"{d}:n={cell[d]['n']}" for d in [7, 8, 9]])
            print(f"    {cell_key}: comeback [{rates}]  [{ns}]")

    print("  EPA summary built")
    for d in [7, 8, 9]:
        s = summary.get(f'deficit_{d}', {})
        print(f"  Down {d}: n={s.get('n',0)}, comeback={s.get('comeback_rate')}, "
              f"scare={s.get('scare_rate')}, avg min={s.get('avg_minutes_left')}")

    return summary


def compute_comeback_aggs(pbp, max_seconds=600):
    q4 = pbp[(pbp['qtr'] == 4) & (pbp['game_seconds_remaining'] <= max_seconds) &
        pbp['game_seconds_remaining'].notna() & pbp['score_differential'].notna() &
        pbp['posteam'].notna()].copy()
    rows = []
    for deficit in [7, 8, 9]:
        sub = q4[q4['score_differential'] == -deficit].copy()
        sub = sub.sort_values('game_seconds_remaining', ascending=False)
        first = sub.groupby(['game_id', 'posteam']).first().reset_index()
        first['deficit'] = deficit
        rows.append(first)
    df = pd.concat(rows, ignore_index=True)
    df['won'] = np.where(df['posteam'] == df['home_team'], df['result'] > 0, df['result'] < 0)
    df['minutes_left'] = df['game_seconds_remaining'] / 60.0
    df['time_bucket'] = pd.cut(df['minutes_left'], bins=[0, 2, 5, 7, 10],
        labels=['0-2', '2-5', '5-7', '7-10'], include_lowest=True)
    def agg(cols):
        return (df.groupby(cols).agg(n=('won', 'count'), wins=('won', 'sum'))
                .reset_index().assign(rate=lambda x: np.where(x['n'] > 0, x['wins'] / x['n'], 0))
                .to_dict('records'))
    return agg(['deficit']), agg(['deficit', 'time_bucket'])


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("NFL 2-Point Conversion Analysis v9")
    print("Direct WP Model + Separate EPA Analysis")
    print("=" * 60)

    # Load WP grid (from R script)
    wp_grid = load_wp_grid()

    # Load PBP data
    pbp = load_pbp(range(2006, 2026))

    print("\nComputing team EPA ratings...")
    ratings = build_team_epa(pbp)

    print("\nComputing conversion rates...")
    conv_rates = compute_conversion_rates(pbp)

    raw_rates, by_time = compute_comeback_aggs(pbp)
    decisions = build_decision_lab(pbp, ratings)
    texture = build_comeback_texture(pbp, ratings)
    epa_summary = build_epa_summary(texture)

    print(f"\n  WP grid: {len(wp_grid):,} rows")
    print(f"  Decisions: {len(decisions):,}")
    print(f"  Texture (kickoff triggers): {len(texture):,}")

    # Quick EV calculation
    print("\n" + "=" * 60)
    print("QUICK EV CHECK (5 min, 3v3 TOs, 48% 2pt, 94% XP)")
    print("=" * 60)
    grid_df = pd.DataFrame(wp_grid)
    for h, label in [(0, "Trailing team AWAY (leading is home)"),
                     (1, "Trailing team HOME (leading is away)")]:
        check = grid_df[(grid_df['game_seconds_remaining'] == 300) &
                        (grid_df['trailing_timeouts'] == 3) &
                        (grid_df['leading_timeouts'] == 3) &
                        (grid_df['trailing_is_home'] == h)]
        if len(check) == 0:
            continue
        wps = {m: check[check['margin'] == m]['leading_wp'].iloc[0] for m in [7, 8, 9]}
        ev_xp = 0.94 * wps[8] + 0.06 * wps[7]
        ev_2pt = 0.48 * wps[9] + 0.52 * wps[7]
        print(f"\n  {label}:")
        print(f"    WP: +7={wps[7]:.4f}  +8={wps[8]:.4f}  +9={wps[9]:.4f}")
        print(f"    EV kick XP:  {ev_xp:.4f}")
        print(f"    EV go for 2: {ev_2pt:.4f}")
        print(f"    Advantage:   {(ev_2pt - ev_xp)*100:+.2f} pp")

    output = {
        'wp_grid': wp_grid,
        'conversion_rates': conv_rates.where(conv_rates.notna(), None).to_dict('records'),
        'comeback': {'raw': raw_rates, 'by_time': by_time},
        'decisions': decisions,
        'texture': texture,
        'epa_summary': epa_summary,
        'seasons': [int(s) for s in sorted(pbp['season'].unique())],
    }

    out_path = Path('dashboard_data.json')
    # Replace NaN/inf with None for valid JSON
    import math
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    output = sanitize(output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=None, default=str)
    mb = out_path.stat().st_size / 1e6
    print(f"\nSaved -> {out_path.resolve()}  ({mb:.1f} MB)")


if __name__ == '__main__':
    main()
