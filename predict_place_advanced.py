#!/usr/bin/env python3
"""
Advanced HKJC Place Prediction System with Sectional Time Analysis
Incorporates:
- Running style analysis (Front-runner, Stalker, Closer)
- Late speed calculation (final 200m/400m splits)
- Pace adaptation metrics
- Position change trends
- Traditional form metrics
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "/workspace/data"
TRIALS_DIR = "/workspace/trials"
HORSE_FORM_DIR = "/workspace/horses/form_records"
HORSE_PROFILE_FILE = "/workspace/horses/profiles/horse_profiles.csv"
ENTRIES_DIR = "/workspace/entries"
OUTPUT_FILE = "/workspace/place_predictions_advanced.csv"

def get_today_date():
    """Get today's date"""
    today = datetime.now().strftime("%Y-%m-%d")
    return today

def load_sectional_times(n_days=180):
    """Load recent sectional times data"""
    today = get_today_date()
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    start_dt = today_dt - timedelta(days=n_days)
    
    all_sectional = []
    
    for year in [2025, 2026]:
        year_dir = os.path.join(DATA_DIR, str(year))
        if not os.path.exists(year_dir):
            continue
        
        for f in glob.glob(os.path.join(year_dir, "sectional_times_*.csv")):
            fname = os.path.basename(f)
            try:
                date_str = fname.replace("sectional_times_", "").replace(".csv", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if start_dt <= file_date <= today_dt:
                    df = pd.read_csv(f)
                    df['file_date'] = date_str
                    all_sectional.append(df)
            except:
                continue
    
    if not all_sectional:
        return pd.DataFrame()
    
    return pd.concat(all_sectional, ignore_index=True)

def load_recent_results(n_days=180):
    """Load recent race results"""
    today = get_today_date()
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    start_dt = today_dt - timedelta(days=n_days)
    
    all_results = []
    
    for year in [2025, 2026]:
        year_dir = os.path.join(DATA_DIR, str(year))
        if not os.path.exists(year_dir):
            continue
        
        for f in glob.glob(os.path.join(year_dir, "results_*.csv")):
            fname = os.path.basename(f)
            try:
                date_str = fname.replace("results_", "").replace(".csv", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if start_dt <= file_date <= today_dt:
                    df = pd.read_csv(f)
                    df['file_date'] = date_str
                    all_results.append(df)
            except:
                continue
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)

def parse_running_positions(running_pos_str):
    """Parse running positions string into list"""
    if not isinstance(running_pos_str, str):
        return []
    
    # Format: "1 2 2 1 1" or similar
    try:
        positions = [int(x) for x in running_pos_str.split()]
        return positions
    except:
        return []

def calculate_running_style(positions):
    """
    Classify running style based on position patterns
    Returns: 'Front-runner', 'Stalker', 'Closer', or 'Mid-pack'
    """
    if len(positions) < 3:
        return 'Unknown'
    
    # Average early position (first half of race)
    mid_point = len(positions) // 2
    early_avg = np.mean(positions[:mid_point]) if mid_point > 0 else 5
    late_avg = np.mean(positions[mid_point:]) if mid_point > 0 else 5
    
    # Final position
    final_pos = positions[-1]
    
    # Classify
    if early_avg <= 3 and final_pos <= 3:
        return 'Front-runner'
    elif early_avg <= 5 and final_pos <= 3:
        return 'Stalker'
    elif early_avg > 6 and final_pos <= 3:
        return 'Closer'
    elif early_avg > 6:
        return 'Deep-closer'
    else:
        return 'Mid-pack'

def calculate_late_speed(sectional_row):
    """Calculate late speed from sectional data"""
    # Try to extract final section time
    # Columns: sec1_time, sec2_time, etc.
    
    late_speeds = []
    
    for i in range(6, 0, -1):
        time_col = f'sec{i}_time'
        if time_col in sectional_row and pd.notna(sectional_row[time_col]):
            time_val = sectional_row[time_col]
            # Parse time string like "11.38" or "11.02    11.19"
            try:
                if isinstance(time_val, str):
                    time_parts = time_val.split()
                    if time_parts:
                        t = float(time_parts[-1])
                        late_speeds.append(t)
                        if len(late_speeds) >= 2:
                            break
            except:
                continue
    
    if len(late_speeds) == 0:
        return None, None
    
    # Last 200m time
    last_200m = late_speeds[0] if late_speeds else None
    
    # Last 400m average (if available)
    if len(late_speeds) >= 2:
        last_400m_avg = (late_speeds[0] + late_speeds[1]) / 2
    else:
        last_400m_avg = last_200m
    
    return last_200m, last_400m_avg

def calculate_position_change(positions):
    """Calculate position change trend"""
    if len(positions) < 2:
        return 0
    
    # Change from first call to finish
    change = positions[0] - positions[-1]
    return change

def calculate_horse_sectional_features(horse_no, sectional_df, results_df, trial_df=None):
    """Calculate advanced features from sectional times and trial performance"""
    
    horse_sectional = sectional_df[sectional_df['horse_no'] == horse_no].tail(10)
    
    # Trial performance features
    trial_features = {}
    if trial_df is not None and not trial_df.empty:
        horse_trials = trial_df[trial_df['horse_name'].str.contains(horse_no, na=False)].tail(5)
        if not horse_trials.empty:
            trial_features = calculate_trial_features(horse_trials)
    
    if horse_sectional.empty:
        base_features = {
            'running_style': 'Unknown',
            'avg_early_pos': 5.0,
            'avg_late_pos': 5.0,
            'position_change': 0.0,
            'late_speed_200m': None,
            'late_speed_400m': None,
            'consistency_score': 0.0,
            'front_run_success': 0.0,
            'closer_success': 0.0
        }
        base_features.update(trial_features)
        return base_features
    
    running_styles = []
    early_positions = []
    late_positions = []
    pos_changes = []
    late_speeds_200m = []
    late_speeds_400m = []
    
    for _, row in horse_sectional.iterrows():
        positions = parse_running_positions(row.get('running_position', ''))
        
        if positions:
            style = calculate_running_style(positions)
            running_styles.append(style)
            
            mid_point = len(positions) // 2
            early_positions.append(np.mean(positions[:mid_point]))
            late_positions.append(np.mean(positions[mid_point:]))
            
            pos_changes.append(calculate_position_change(positions))
        
        # Late speed
        last_200m, last_400m = calculate_late_speed(row)
        if last_200m:
            late_speeds_200m.append(last_200m)
        if last_400m:
            late_speeds_400m.append(last_400m)
    
    # Calculate aggregates
    style_counts = pd.Series(running_styles).value_counts()
    dominant_style = style_counts.index[0] if len(style_counts) > 0 else 'Unknown'
    
    avg_early = np.mean(early_positions) if early_positions else 5.0
    avg_late = np.mean(late_positions) if late_positions else 5.0
    avg_pos_change = np.mean(pos_changes) if pos_changes else 0.0
    
    avg_late_speed_200m = np.mean(late_speeds_200m) if late_speeds_200m else None
    avg_late_speed_400m = np.mean(late_speeds_400m) if late_speeds_400m else None
    
    # Consistency score (lower std in positions = more consistent)
    if len(late_positions) > 1:
        consistency = 1.0 / (1.0 + np.std(late_positions))
    else:
        consistency = 0.5
    
    # Front-run success rate
    front_runs = [i for i, s in enumerate(running_styles) if s in ['Front-runner', 'Stalker']]
    front_wins = len([i for i in front_runs if i < len(horse_sectional) and 
                      horse_sectional.iloc[i]['finish_pos'] <= 3]) if front_runs else 0
    front_success = front_wins / len(front_runs) if front_runs else 0.0
    
    # Closer success rate
    closer_runs = [i for i, s in enumerate(running_styles) if s in ['Closer', 'Deep-closer']]
    closer_wins = len([i for i in closer_runs if i < len(horse_sectional) and 
                       horse_sectional.iloc[i]['finish_pos'] <= 3]) if closer_runs else 0
    closer_success = closer_wins / len(closer_runs) if closer_runs else 0.0
    
    return {
        'running_style': dominant_style,
        'avg_early_pos': avg_early,
        'avg_late_pos': avg_late,
        'position_change': avg_pos_change,
        'late_speed_200m': avg_late_speed_200m,
        'late_speed_400m': avg_late_speed_400m,
        'consistency_score': consistency,
        'front_run_success': front_success,
        'closer_success': closer_success,
        **trial_features
    }

def calculate_trial_features(trial_df):
    """Calculate features from trial performance"""
    if trial_df.empty:
        return {
            'trial_runs': 0,
            'trial_avg_position': 5.0,
            'trial_best_position': 5.0,
            'trial_place_rate': 0.0,
            'trial_commentary_positive': 0.0
        }
    
    # Parse running positions from trials
    positions = []
    for _, row in trial_df.iterrows():
        pos_str = row.get('running_position', '')
        if isinstance(pos_str, str):
            try:
                pos_list = [int(x) for x in pos_str.split()]
                if pos_list:
                    positions.append(pos_list[-1])  # Final position
            except:
                continue
    
    # Calculate trial metrics
    trial_runs = len(trial_df)
    avg_position = np.mean(positions) if positions else 5.0
    best_position = min(positions) if positions else 5.0
    place_count = len([p for p in positions if p <= 3])
    place_rate = place_count / trial_runs if trial_runs > 0 else 0.0
    
    # Analyze commentary for positive signals
    positive_keywords = ['勁', '佳', '好', '優', '勝', '凌厲', '輕鬆', '出色', '滿意', '進步']
    commentary_scores = []
    for _, row in trial_df.iterrows():
        commentary = row.get('commentary', '')
        if isinstance(commentary, str):
            score = sum(1 for kw in positive_keywords if kw in commentary)
            commentary_scores.append(score)
    
    avg_commentary_score = np.mean(commentary_scores) if commentary_scores else 0.0
    max_commentary_score = max(commentary_scores) if commentary_scores else 0.0
    commentary_positive_rate = avg_commentary_score / 5.0  # Normalize
    
    return {
        'trial_runs': trial_runs,
        'trial_avg_position': avg_position,
        'trial_best_position': best_position,
        'trial_place_rate': place_rate,
        'trial_commentary_positive': commentary_positive_rate
    }

def calculate_horse_features(horse_no, results_df, sectional_df, reference_date=None):
    """Calculate comprehensive features for a horse"""
    
    # Load form from files
    form_file = os.path.join(HORSE_FORM_DIR, f"form_{horse_no}.csv")
    if os.path.exists(form_file):
        try:
            form_df = pd.read_csv(form_file)
        except:
            form_df = pd.DataFrame()
    else:
        form_df = pd.DataFrame()
    
    if form_df.empty:
        base_features = {
            'horse_no': horse_no,
            'recent_win_rate': 0.0,
            'recent_place_rate': 0.0,
            'avg_finish_pos': 8.0,
            'recent_runs': 0,
            'best_recent_pos': 8.0,
            'odds_avg': 10.0
        }
    else:
        form_df = form_df.sort_values('race_index', ascending=False).head(10)
        total_runs = len(form_df)
        
        form_df['place_numeric'] = pd.to_numeric(form_df['place'], errors='coerce').fillna(99)
        form_df['win_odds_numeric'] = pd.to_numeric(
            form_df.get('win_odds', pd.Series([10]*len(form_df))), 
            errors='coerce'
        ).fillna(10.0)
        
        wins = len(form_df[form_df['place_numeric'] == 1])
        win_rate = wins / total_runs if total_runs > 0 else 0.0
        
        places = len(form_df[form_df['place_numeric'] <= 3])
        place_rate = places / total_runs if total_runs > 0 else 0.0
        
        avg_pos = form_df['place_numeric'].mean()
        best_pos = form_df['place_numeric'].min()
        avg_odds = form_df['win_odds_numeric'].mean()
        
        base_features = {
            'horse_no': horse_no,
            'recent_win_rate': win_rate,
            'recent_place_rate': place_rate,
            'avg_finish_pos': avg_pos,
            'recent_runs': total_runs,
            'best_recent_pos': best_pos,
            'odds_avg': avg_odds
        }
    
    # Load trial data
    trial_df = load_trial_data()
    
    # Add sectional-based features (including trial data)
    sectional_features = calculate_horse_sectional_features(horse_no, sectional_df, results_df, trial_df)
    base_features.update(sectional_features)
    
    return base_features

def load_trial_data():
    """Load trial results data"""
    trial_file = os.path.join(TRIALS_DIR, "trial_results.csv")
    if os.path.exists(trial_file):
        try:
            df = pd.read_csv(trial_file)
            return df
        except Exception as e:
            print(f"Warning: Could not load trial data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_jockey_stats(jockey_name, results_df):
    """Calculate jockey statistics"""
    if not isinstance(jockey_name, str) or not jockey_name.strip():
        return {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
    
    jockey_results = results_df[results_df['jockey'] == jockey_name]
    
    if len(jockey_results) == 0:
        return {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
    
    rides = jockey_results.groupby(['date', 'race_no']).first().reset_index()
    total_rides = len(rides)
    
    wins = len(rides[rides['place'] == 1])
    places = len(rides[rides['place'] <= 3])
    
    return {
        'jockey_win_rate': wins / total_rides if total_rides > 0 else 0.0,
        'jockey_place_rate': places / total_rides if total_rides > 0 else 0.0
    }

def get_trainer_stats(trainer_name, results_df):
    """Calculate trainer statistics"""
    if not isinstance(trainer_name, str) or not trainer_name.strip():
        return {'trainer_win_rate': 0.0, 'trainer_place_rate': 0.0}
    
    trainer_results = results_df[results_df['trainer'] == trainer_name]
    
    if len(trainer_results) == 0:
        return {'trainer_win_rate': 0.0, 'trainer_place_rate': 0.0}
    
    runners = trainer_results.groupby(['date', 'race_no']).first().reset_index()
    total_runners = len(runners)
    
    wins = len(runners[runners['place'] == 1])
    places = len(runners[runners['place'] <= 3])
    
    return {
        'trainer_win_rate': wins / total_runners if total_runners > 0 else 0.0,
        'trainer_place_rate': places / total_runners if total_runners > 0 else 0.0
    }

def calculate_advanced_place_score(features, race_conditions=None):
    """
    Calculate advanced place score incorporating running style, sectional data, and trial performance
    """
    score = 0.0
    
    # Traditional form metrics (45% weight)
    score += features.get('recent_place_rate', 0) * 22
    score += features.get('recent_win_rate', 0) * 13
    
    avg_pos = features.get('avg_finish_pos', 8)
    pos_score = max(0, (10 - avg_pos) / 10)
    score += pos_score * 10
    
    # Trial performance metrics (15% weight)
    trial_place_rate = features.get('trial_place_rate', 0)
    if trial_place_rate > 0:
        score += trial_place_rate * 10
    
    trial_commentary = features.get('trial_commentary_positive', 0)
    score += trial_commentary * 5
    
    # Running style metrics (20% weight)
    consistency = features.get('consistency_score', 0.5)
    score += consistency * 8
    
    pos_change = features.get('position_change', 0)
    if pos_change > 0:  # Positive position change is good
        score += min(pos_change, 5) * 2.5
    
    # Late speed metrics (12% weight)
    late_speed = features.get('late_speed_200m')
    if late_speed and late_speed < 12.0:  # Good late speed
        speed_score = max(0, (12.5 - late_speed) / 2.5)
        score += speed_score * 8
    
    # Early position advantage (8% weight)
    early_pos = features.get('avg_early_pos', 5)
    if early_pos <= 4:
        score += (5 - early_pos) * 2
    
    # Jockey/trainer stats (10% weight)
    score += features.get('jockey_place_rate', 0) * 5
    score += features.get('trainer_place_rate', 0) * 3
    
    return score

def load_entries_for_date(date_str):
    """Load entry list for a specific date"""
    entry_file = os.path.join(ENTRIES_DIR, f"entries_{date_str}.txt")
    
    if not os.path.exists(entry_file):
        entry_file = os.path.join(ENTRIES_DIR, "today_entries.txt")
    
    if not os.path.exists(entry_file):
        return None
    
    horses = []
    with open(entry_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line:
                horses.append(line)
    
    return horses

def get_race_entries(entries_list, results_df):
    """Group entries into races"""
    if not entries_list:
        return {}
    
    horses_per_race = 11
    races = {}
    
    for i, horse_no in enumerate(entries_list):
        race_no = (i // horses_per_race) + 1
        if race_no not in races:
            races[race_no] = []
        races[race_no].append(horse_no)
    
    return races

def predict_for_race_advanced(race_no, horse_nos, results_df, sectional_df):
    """Predict top 4 place candidates using advanced model"""
    predictions = []
    
    for horse_no in horse_nos:
        features = calculate_horse_features(horse_no, results_df, sectional_df)
        
        jockey_stats = {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
        trainer_stats = {'trainer_win_rate': 0.0, 'trainer_place_rate': 0.0}
        
        recent_horse_results = results_df[results_df['horse_no'] == horse_no]
        if not recent_horse_results.empty:
            last_race = recent_horse_results.iloc[-1]
            jockey_name = last_race.get('jockey', '')
            trainer_name = last_race.get('trainer', '')
            
            jockey_stats = get_jockey_stats(jockey_name, results_df)
            trainer_stats = get_trainer_stats(trainer_name, results_df)
        
        features.update(jockey_stats)
        features.update(trainer_stats)
        
        place_score = calculate_advanced_place_score(features)
        
        predictions.append({
            'race_no': race_no,
            'horse_no': horse_no,
            'place_score': place_score,
            'recent_place_rate': features.get('recent_place_rate', 0),
            'recent_win_rate': features.get('recent_win_rate', 0),
            'avg_finish_pos': features.get('avg_finish_pos', 0),
            'recent_runs': features.get('recent_runs', 0),
            'running_style': features.get('running_style', 'Unknown'),
            'avg_early_pos': features.get('avg_early_pos', 0),
            'position_change': features.get('position_change', 0),
            'late_speed_200m': features.get('late_speed_200m'),
            'consistency_score': features.get('consistency_score', 0),
            'jockey_place_rate': features.get('jockey_place_rate', 0),
            'trainer_place_rate': features.get('trainer_place_rate', 0),
            'trial_runs': features.get('trial_runs', 0),
            'trial_place_rate': features.get('trial_place_rate', 0),
            'trial_commentary_positive': features.get('trial_commentary_positive', 0)
        })
    
    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        return predictions_df
    
    predictions_df = predictions_df.sort_values('place_score', ascending=False)
    top4 = predictions_df.head(4)
    
    return top4

def main():
    print("=" * 70)
    print("HKJC Advanced Place Prediction System")
    print("With Sectional Time & Running Style Analysis")
    print("=" * 70)
    
    today = get_today_date()
    print(f"\nPrediction Date: {today}")
    
    # Load data
    print("\nLoading recent race results...")
    results_df = load_recent_results(n_days=180)
    print(f"Loaded {len(results_df)} race records")
    
    print("\nLoading sectional times data...")
    sectional_df = load_sectional_times(n_days=180)
    print(f"Loaded {len(sectional_df)} sectional time records")
    
    print("\nLoading today's entries...")
    entries = load_entries_for_date(today)
    
    if not entries:
        print("No entries found for today. Using latest available...")
        entry_files = sorted(glob.glob(os.path.join(ENTRIES_DIR, "entries_*.txt")))
        if entry_files:
            latest_entry = entry_files[-1]
            date_str = latest_entry.split('_')[-1].replace('.txt', '')
            entries = load_entries_for_date(date_str)
            today = date_str
            print(f"Using entries from {today}")
    
    if not entries:
        print("ERROR: No entries available!")
        return
    
    print(f"Found {len(entries)} horses entered")
    
    races = get_race_entries(entries, results_df)
    print(f"Estimated {len(races)} races")
    
    all_predictions = []
    
    print("\nGenerating predictions with sectional analysis...")
    for race_no in sorted(races.keys()):
        horse_nos = races[race_no]
        top4 = predict_for_race_advanced(race_no, horse_nos, results_df, sectional_df)
        
        if not top4.empty:
            all_predictions.append(top4)
            print(f"  Race {race_no}: {len(horse_nos)} horses -> Top 4 predicted")
    
    if not all_predictions:
        print("No predictions generated!")
        return
    
    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df['prediction_rank'] = final_df.groupby('race_no').cumcount() + 1
    
    cols = ['race_no', 'prediction_rank', 'horse_no', 'place_score',
            'recent_place_rate', 'recent_win_rate', 'avg_finish_pos',
            'running_style', 'avg_early_pos', 'position_change',
            'late_speed_200m', 'consistency_score',
            'recent_runs', 'jockey_place_rate', 'trainer_place_rate',
            'trial_runs', 'trial_place_rate', 'trial_commentary_positive']
    final_df = final_df[cols]
    
    numeric_cols = ['place_score', 'recent_place_rate', 'recent_win_rate',
                    'avg_finish_pos', 'avg_early_pos', 'position_change',
                    'late_speed_200m', 'consistency_score',
                    'jockey_place_rate', 'trainer_place_rate',
                    'trial_runs', 'trial_place_rate', 'trial_commentary_positive']
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='ignore')
            if pd.api.types.is_numeric_dtype(final_df[col]):
                final_df[col] = final_df[col].round(4)
    
    output_path = OUTPUT_FILE
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Predictions saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    
    for race_no in sorted(final_df['race_no'].unique()):
        race_preds = final_df[final_df['race_no'] == race_no]
        print(f"\nRace {race_no} - Top 4 Place Candidates:")
        for _, row in race_preds.iterrows():
            style_info = f"[{row['running_style']}]" if row['running_style'] != 'Unknown' else ""
            late_speed_info = f"{row['late_speed_200m']:.2f}s" if pd.notna(row['late_speed_200m']) else "N/A"
            trial_info = f"Trials: {int(row['trial_runs'])}" if row['trial_runs'] > 0 else "No trials"
            print(f"  #{int(row['prediction_rank'])} Horse {row['horse_no']} "
                  f"(Score: {row['place_score']:.2f}, {style_info})")
            print(f"      Place Rate: {row['recent_place_rate']:.1%}, "
                  f"Early Pos: {row['avg_early_pos']:.1f}, "
                  f"Late Speed: {late_speed_info}, {trial_info}")
    
    print("\n" + "=" * 70)
    print(f"Total predictions: {len(final_df)} horses across {len(final_df['race_no'].unique())} races")
    print("=" * 70)

if __name__ == "__main__":
    main()
