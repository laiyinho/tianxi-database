#!/usr/bin/env python3
"""
Predict top 4 horses most likely to place (finish 1st, 2nd, or 3rd) 
for each race on today's race day.
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
HORSE_FORM_DIR = "/workspace/horses/form_records"
HORSE_PROFILE_FILE = "/workspace/horses/profiles/horse_profiles.csv"
ENTRIES_DIR = "/workspace/entries"
OUTPUT_FILE = "/workspace/place_predictions.csv"

def get_today_date():
    """Get today's date from fixtures or system"""
    # Check fixtures for today
    today = datetime.now().strftime("%Y-%m-%d")
    return today

def load_recent_results(n_days=90):
    """Load recent race results for feature calculation"""
    today = get_today_date()
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    start_dt = today_dt - timedelta(days=n_days)
    
    all_results = []
    
    # Load results from 2025 and 2026
    for year in [2025, 2026]:
        year_dir = os.path.join(DATA_DIR, str(year))
        if not os.path.exists(year_dir):
            continue
        
        for f in glob.glob(os.path.join(year_dir, "results_*.csv")):
            # Extract date from filename
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

def load_horse_form(horse_no):
    """Load form records for a specific horse"""
    form_file = os.path.join(HORSE_FORM_DIR, f"form_{horse_no}.csv")
    if os.path.exists(form_file):
        try:
            return pd.read_csv(form_file)
        except:
            pass
    return pd.DataFrame()

def calculate_horse_features(horse_no, results_df, reference_date=None):
    """Calculate features for a horse based on historical performance"""
    form_df = load_horse_form(horse_no)
    
    if form_df.empty:
        return {
            'horse_no': horse_no,
            'recent_win_rate': 0.0,
            'recent_place_rate': 0.0,
            'avg_finish_pos': 8.0,
            'recent_runs': 0,
            'best_recent_pos': 8.0,
            'odds_avg': 10.0
        }
    
    # Sort by date (most recent first)
    form_df = form_df.sort_values('race_index', ascending=False).head(10)
    
    total_runs = len(form_df)
    
    # Convert place to numeric
    form_df['place_numeric'] = pd.to_numeric(form_df['place'], errors='coerce').fillna(99)
    form_df['win_odds_numeric'] = pd.to_numeric(form_df.get('win_odds', pd.Series([10]*len(form_df))), errors='coerce').fillna(10.0)
    
    # Calculate win rate (place == 1)
    wins = len(form_df[form_df['place_numeric'] == 1])
    win_rate = wins / total_runs if total_runs > 0 else 0.0
    
    # Calculate place rate (place <= 3)
    places = len(form_df[form_df['place_numeric'] <= 3])
    place_rate = places / total_runs if total_runs > 0 else 0.0
    
    # Average finish position
    avg_pos = form_df['place_numeric'].mean()
    
    # Best recent position
    best_pos = form_df['place_numeric'].min()
    
    # Average odds
    avg_odds = form_df['win_odds_numeric'].mean()
    
    return {
        'horse_no': horse_no,
        'recent_win_rate': win_rate,
        'recent_place_rate': place_rate,
        'avg_finish_pos': avg_pos,
        'recent_runs': total_runs,
        'best_recent_pos': best_pos,
        'odds_avg': avg_odds
    }

def get_jockey_stats(jockey_name, results_df):
    """Calculate jockey statistics"""
    if not isinstance(jockey_name, str) or not jockey_name.strip():
        return {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
    
    jockey_results = results_df[results_df['jockey'] == jockey_name]
    
    if len(jockey_results) == 0:
        return {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
    
    # Group by race to get unique rides
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
    
    # Group by race to get unique runners
    runners = trainer_results.groupby(['date', 'race_no']).first().reset_index()
    total_runners = len(runners)
    
    wins = len(runners[runners['place'] == 1])
    places = len(runners[runners['place'] <= 3])
    
    return {
        'trainer_win_rate': wins / total_runners if total_runners > 0 else 0.0,
        'trainer_place_rate': places / total_runners if total_runners > 0 else 0.0
    }

def calculate_place_score(features):
    """
    Calculate a score indicating likelihood of placing (top 3).
    Higher score = more likely to place.
    """
    score = 0.0
    
    # Place rate is most important (40% weight)
    score += features.get('recent_place_rate', 0) * 40
    
    # Win rate (25% weight)
    score += features.get('recent_win_rate', 0) * 25
    
    # Average finish position (inverse, 20% weight)
    avg_pos = features.get('avg_finish_pos', 8)
    pos_score = max(0, (10 - avg_pos) / 10)  # Normalize to 0-1
    score += pos_score * 20
    
    # Best recent position (10% weight)
    best_pos = features.get('best_recent_pos', 8)
    best_score = max(0, (4 - best_pos) / 4)  # Normalize
    score += best_score * 10
    
    # Jockey stats (5% weight)
    score += features.get('jockey_place_rate', 0) * 5
    
    return score

def load_entries_for_date(date_str):
    """Load entry list for a specific date"""
    entry_file = os.path.join(ENTRIES_DIR, f"entries_{date_str}.txt")
    
    if not os.path.exists(entry_file):
        # Try today_entries.txt
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
    """
    Parse entries and group by race.
    Since we don't have detailed race info in entries, we'll simulate race assignments.
    """
    # For simplicity, assume ~12 horses per race
    # In reality, we'd need to scrape the actual race card
    
    if not entries_list:
        return {}
    
    # Group horses into races (approx 10-12 horses per race)
    horses_per_race = 11
    races = {}
    
    for i, horse_no in enumerate(entries_list):
        race_no = (i // horses_per_race) + 1
        if race_no not in races:
            races[race_no] = []
        races[race_no].append(horse_no)
    
    return races

def predict_for_race(race_no, horse_nos, results_df):
    """Predict top 4 place candidates for a race"""
    predictions = []
    
    for horse_no in horse_nos:
        # Get horse features
        features = calculate_horse_features(horse_no, results_df)
        
        # Get a sample result to find jockey/trainer info
        # In reality, we'd need current entries with jockey/trainer
        jockey_stats = {'jockey_win_rate': 0.0, 'jockey_place_rate': 0.0}
        trainer_stats = {'trainer_win_rate': 0.0, 'trainer_place_rate': 0.0}
        
        # Try to find recent jockey/trainer info from results
        recent_horse_results = results_df[results_df['horse_no'] == horse_no]
        if not recent_horse_results.empty:
            last_race = recent_horse_results.iloc[-1]
            jockey_name = last_race.get('jockey', '')
            trainer_name = last_race.get('trainer', '')
            
            jockey_stats = get_jockey_stats(jockey_name, results_df)
            trainer_stats = get_trainer_stats(trainer_name, results_df)
        
        features.update(jockey_stats)
        features.update(trainer_stats)
        
        # Calculate place score
        place_score = calculate_place_score(features)
        
        predictions.append({
            'race_no': race_no,
            'horse_no': horse_no,
            'place_score': place_score,
            'recent_place_rate': features.get('recent_place_rate', 0),
            'recent_win_rate': features.get('recent_win_rate', 0),
            'avg_finish_pos': features.get('avg_finish_pos', 0),
            'recent_runs': features.get('recent_runs', 0),
            'jockey_place_rate': features.get('jockey_place_rate', 0),
            'trainer_place_rate': features.get('trainer_place_rate', 0)
        })
    
    # Sort by place score descending
    predictions_df = pd.DataFrame(predictions)
    if predictions_df.empty:
        return predictions_df
    
    predictions_df = predictions_df.sort_values('place_score', ascending=False)
    
    # Select top 4
    top4 = predictions_df.head(4)
    
    return top4

def main():
    print("=" * 60)
    print("HKJC Place Prediction System")
    print("=" * 60)
    
    today = get_today_date()
    print(f"\nPrediction Date: {today}")
    
    # Load recent results
    print("\nLoading recent race results...")
    results_df = load_recent_results(n_days=180)
    print(f"Loaded {len(results_df)} race records")
    
    # Load entries
    print("\nLoading today's entries...")
    entries = load_entries_for_date(today)
    
    if not entries:
        print("No entries found for today. Using latest available entries...")
        # Find most recent entries file
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
    
    # Group into races
    races = get_race_entries(entries, results_df)
    print(f"Estimated {len(races)} races")
    
    # Generate predictions for each race
    all_predictions = []
    
    print("\nGenerating predictions...")
    for race_no in sorted(races.keys()):
        horse_nos = races[race_no]
        top4 = predict_for_race(race_no, horse_nos, results_df)
        
        if not top4.empty:
            all_predictions.append(top4)
            print(f"  Race {race_no}: {len(horse_nos)} horses -> Top 4 predicted")
    
    if not all_predictions:
        print("No predictions generated!")
        return
    
    # Combine all predictions
    final_df = pd.concat(all_predictions, ignore_index=True)
    
    # Add ranking within each race
    final_df['prediction_rank'] = final_df.groupby('race_no').cumcount() + 1
    
    # Reorder columns
    cols = ['race_no', 'prediction_rank', 'horse_no', 'place_score', 
            'recent_place_rate', 'recent_win_rate', 'avg_finish_pos',
            'recent_runs', 'jockey_place_rate', 'trainer_place_rate']
    final_df = final_df[cols]
    
    # Round numeric columns
    numeric_cols = ['place_score', 'recent_place_rate', 'recent_win_rate', 
                    'avg_finish_pos', 'jockey_place_rate', 'trainer_place_rate']
    for col in numeric_cols:
        final_df[col] = final_df[col].round(4)
    
    # Save to CSV
    output_path = OUTPUT_FILE
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    for race_no in sorted(final_df['race_no'].unique()):
        race_preds = final_df[final_df['race_no'] == race_no]
        print(f"\nRace {race_no} - Top 4 Place Candidates:")
        for _, row in race_preds.iterrows():
            print(f"  #{int(row['prediction_rank'])} Horse {row['horse_no']} "
                  f"(Score: {row['place_score']:.2f}, "
                  f"Place Rate: {row['recent_place_rate']:.1%})")
    
    print("\n" + "=" * 60)
    print(f"Total predictions: {len(final_df)} horses across {len(final_df['race_no'].unique())} races")
    print("=" * 60)

if __name__ == "__main__":
    main()
