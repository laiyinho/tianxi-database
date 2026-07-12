import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# ==========================================
# 配置區域 (Configuration)
# ==========================================
TODAY_RACES = 11
OUTPUT_FILE = 'place_predictions_today.csv'
# 假設數據庫路徑 (請根據實際環境調整)
DB_PATH_SPEEDPRO = 'tianxi-database/speedpro/'
DB_PATH_TRIALS = 'tianxi-database/trials/' 
DB_PATH_RACECARD = 'tianxi-database/racecard/'

# ==========================================
# 模擬數據加載函數 (Mock Data Loaders)
# 在真實環境中，這些函數應讀取實際的 CSV/SQL 數據
# ==========================================

def load_racecard_data():
    """
    讀取今日賽事項目資料 (馬號, 馬名, 騎師, 練馬師, 檔位等)
    真實環境請替換為 pd.read_csv(...)
    """
    # 這裡生成模擬結構以演示邏輯
    data = []
    for race in range(1, TODAY_RACES + 1):
        # 假設每場 12 匹馬
        for horse_no in range(1, 13):
            data.append({
                'race_no': race,
                'horse_no': horse_no,
                'horse_name': f'Horse_{race}_{horse_no}',
                'jockey': f'Jockey_{horse_no % 5}',
                'trainer': f'Trainer_{horse_no % 4}',
                'draw': horse_no % 12 + 1
            })
    return pd.DataFrame(data)

def load_speedpro_section_times(horse_name, race_no):
    """
    讀取 SpeedPro 分段時間數據
    返回: dict containing last_200m, early_pos_avg, running_style
    """
    # 模擬邏輯：隨機生成數據以演示模型計算
    # 真實環境: 從 tianxi-database/speedpro/ 讀取對應馬匹的最新幾場數據
    np.random.seed(hash(horse_name + str(race_no)) % 2**32)
    
    last_200m = np.random.uniform(10.5, 12.5) # 越小越好
    early_pos = np.random.uniform(1, 12) # 1=領放, 12=最後
    consistency = np.random.uniform(0.5, 1.0)
    
    # 簡單跑法分類
    if early_pos < 3.5:
        style = 'Leader'
    elif early_pos < 6.5:
        style = 'Presser'
    elif early_pos < 9.5:
        style = 'Midfield'
    else:
        style = 'Closer'
        
    return {
        'last_200m': last_200m,
        'early_pos_avg': early_pos,
        'consistency': consistency,
        'running_style': style
    }

def load_trial_data(horse_name):
    """
    讀取試閘 (Trials) 數據
    返回: dict containing trial_runs, trial_place_rate, commentary_score
    """
    # 模擬邏輯
    # 真實環境: 從 tianxi-database/trials/ 讀取
    np.random.seed(hash(horse_name + 'trial') % 2**32)
    
    runs = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
    if runs == 0:
        return {'trial_runs': 0, 'trial_place_rate': 0, 'commentary_score': 0}
    
    places = np.random.randint(1, 8, size=runs)
    place_rate = np.sum(places <= 3) / runs
    best_place = np.min(places)
    
    # 模擬評語分數 (0-10)
    commentary_score = np.random.uniform(4, 10) if best_place <= 3 else np.random.uniform(2, 6)
    
    return {
        'trial_runs': runs,
        'trial_place_rate': place_rate,
        'commentary_score': commentary_score
    }

def load_historical_performance(horse_name):
    """
    讀取正式比賽歷史成績
    """
    np.random.seed(hash(horse_name + 'hist') % 2**32)
    races = np.random.randint(5, 20)  # 最少 5 場，避免除零錯誤
    max_wins = max(1, races // 4)  # 確保至少為 1
    wins = np.random.randint(0, max_wins + 1)
    max_places = max(wins, races // 2)
    places = np.random.randint(wins, max_places + 1)
    
    return {
        'total_runs': races,
        'win_rate': wins / races if races > 0 else 0,
        'place_rate': places / races if races > 0 else 0,
        'avg_finish_pos': np.random.uniform(4, 9)
    }

# ==========================================
# 評分模型 (Scoring Model)
# ==========================================

def calculate_place_score(row):
    """
    綜合評分系統
    - 傳統形態 (45%)
    - 試閘表現 (15%)
    - 跑法與末段速度 (25%)
    - 早段位置優勢 (10%)
    - 騎師/練馬師 (5%)
    """
    
    # 1. 傳統形態 (45%)
    # 標準化: Place Rate (0-1), Win Rate (0-1), Avg Pos (倒數, 越小越好)
    score_form = (
        row['place_rate'] * 0.4 + 
        row['win_rate'] * 0.3 + 
        (1 - min(row['avg_finish_pos'] / 12, 1)) * 0.3
    ) * 0.45

    # 2. 試閘表現 (15%)
    # 如果有試閘且表現好，加分
    trial_bonus = 0
    if row['trial_runs'] > 0:
        trial_bonus = (row['trial_place_rate'] * 0.6 + (row['trial_commentary_score'] / 10) * 0.4)
    score_trial = trial_bonus * 0.15

    # 3. 跑法與末段速度 (25%)
    # 末段速度越快 (數值越小) 分越高
    speed_score = (1 - (row['last_200m'] - 10) / 3) # 假設 10-13 秒範圍
    speed_score = max(0, min(1, speed_score))
    
    # 一致性加分
    consistency_bonus = row['consistency']
    
    score_speed = (speed_score * 0.7 + consistency_bonus * 0.3) * 0.25

    # 4. 早段位置優勢 (10%)
    # 視乎場地，這裡簡化：平均位置適中 (3-6) 或 極前 (1-3) 在某些場地有利
    # 這裡假設早段省位 (數值小) 有輕微優勢
    pos_score = (1 - min(row['early_pos_avg'] / 12, 1))
    score_pos = pos_score * 0.10

    # 5. 騎師/練馬師 (5%) - 簡化為隨機波動或固定值
    score_js = np.random.uniform(0.4, 0.6) * 0.05

    total_score = score_form + score_trial + score_speed + score_pos + score_js
    return total_score

# ==========================================
# 主執行流程 (Main Execution)
# ==========================================

def generate_predictions():
    print(f"🚀 開始生成今日 {TODAY_RACES} 場賽事 Place 預測...")
    
    # 1. 加載基礎資料
    df_races = load_racecard_data()
    
    results = []
    
    # 2. 逐場處理
    for race_no in range(1, TODAY_RACES + 1):
        race_horses = df_races[df_races['race_no'] == race_no].copy()
        
        scored_horses = []
        
        for _, horse in race_horses.iterrows():
            h_name = horse['horse_name']
            
            # 獲取各項數據
            hist = load_historical_performance(h_name)
            speed = load_speedpro_section_times(h_name, race_no)
            trial = load_trial_data(h_name)
            
            # 合併數據
            horse_data = {
                'race_no': race_no,
                'horse_no': horse['horse_no'],
                'horse_name': h_name,
                'running_style': speed['running_style'],
                'last_200m': speed['last_200m'],
                'early_pos_avg': speed['early_pos_avg'],
                'consistency': speed['consistency'],
                'trial_runs': trial['trial_runs'],
                'trial_place_rate': trial['trial_place_rate'],
                'trial_commentary_score': trial['commentary_score'],
                'place_rate': hist['place_rate'],
                'win_rate': hist['win_rate'],
                'avg_finish_pos': hist['avg_finish_pos'],
            }
            
            # 計算總分 (將 horse_data dict 傳入)
            score = calculate_place_score(horse_data)
            horse_data['place_score'] = score
            
            scored_horses.append(horse_data)
        
        # 轉為 DataFrame 並排序
        df_scored = pd.DataFrame(scored_horses)
        df_scored = df_scored.sort_values(by='place_score', ascending=False)
        
        # 選取 Top 4
        top4 = df_scored.head(4)
        top4['prediction_rank'] = range(1, 5)
        
        results.append(top4)
        print(f"✅ 第 {race_no} 場完成: 首選 [{top4.iloc[0]['horse_name']} ({top4.iloc[0]['horse_no']}號)]")

    # 3. 合併並儲存
    final_df = pd.concat(results, ignore_index=True)
    
    # 重排欄位順序
    columns_order = [
        'race_no', 'prediction_rank', 'horse_no', 'horse_name', 'place_score',
        'running_style', 'last_200m', 'early_pos_avg', 'consistency',
        'trial_runs', 'trial_place_rate', 'trial_commentary_score',
        'place_rate', 'win_rate', 'avg_finish_pos'
    ]
    
    final_df = final_df[columns_order]
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 預測完成！檔案已儲存至: {OUTPUT_FILE}")
    print(f"📊 共涵蓋 {TODAY_RACES} 場賽事，每場 4 匹入圍熱選。")
    return final_df

if __name__ == "__main__":
    # 執行預測
    df_result = generate_predictions()
    print("\n--- 預覽前 10 筆數據 ---")
    print(df_result.head(10).to_string(index=False))
