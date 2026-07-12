import pandas as pd
import numpy as np
import random

# 設定隨機種子以確保結果可重現 (實際使用時可移除)
np.random.seed(42)
random.seed(42)

def generate_horse_data(race_no, num_horses):
    """模擬生成馬匹數據，包含 SpeedPro, Section Time, Trials 等指標"""
    horses = []
    
    # 模擬跑法類型 (1: 領放, 2: 跟隨, 3: 中游, 4: 後上, 5: 深後上)
    running_styles = {1: 'Lead', 2: 'Presser', 3: 'Midfield', 4: 'Closer', 5: 'Deep Closer'}
    
    for i in range(1, num_horses + 1):
        horse_no = i
        
        # --- 基礎形態數據 ---
        recent_runs = np.random.randint(3, 15)
        win_rate = np.random.beta(2, 5) * 0.4  # 0-40%
        place_rate = win_rate + np.random.beta(2, 3) * 0.3  # 通常比勝率高
        avg_pos = np.random.normal(5, 2.5)
        avg_pos = max(1, min(14, avg_pos))
        
        # --- SpeedPro & Section Time 衍生數據 ---
        # 最後 200m 速度 (相對值，越高越好)
        last_200_speed = np.random.normal(50, 8) 
        # 早段平均位置 (1-14, 越小越前)
        early_pos_avg = np.random.normal(6, 3)
        early_pos_avg = max(1, min(14, early_pos_avg))
        
        # 跑法風格 (根據早段位置和最後速度推斷)
        if early_pos_avg < 3.5:
            style_code = 1 # Lead
        elif early_pos_avg < 6:
            style_code = 2 # Presser
        elif early_pos_avg < 9:
            style_code = 3 # Midfield
        elif last_200_speed > 55:
            style_code = 5 # Deep Closer
        else:
            style_code = 4 # Closer
            
        style_name = running_styles[style_code]
        
        # 一致性分數 (基於位置變化標準差)
        consistency = np.random.uniform(0.3, 0.95)
        
        # --- 試閘 (Trials) 數據 ---
        has_trial = np.random.choice([True, False], p=[0.3, 0.7]) # 30% 馬有試閘
        trial_runs = 0
        trial_place_rate = 0.0
        trial_commentary_score = 0.0
        
        if has_trial:
            trial_runs = np.random.randint(1, 4)
            # 試閘入位率通常較高
            trial_place_rate = np.random.uniform(0.4, 1.0)
            # 評語分數 (0-10)
            trial_commentary_score = np.random.uniform(4, 10) if np.random.random() > 0.3 else np.random.uniform(2, 5)
        
        # --- 騎師/練馬師 ---
        jockey_trainer_bonus = np.random.uniform(0.8, 1.2)
        
        horses.append({
            'race_no': race_no,
            'horse_no': horse_no,
            'win_rate': win_rate,
            'place_rate': place_rate,
            'avg_pos': avg_pos,
            'last_200_speed': last_200_speed,
            'early_pos_avg': early_pos_avg,
            'running_style': style_name,
            'consistency': consistency,
            'trial_runs': trial_runs,
            'trial_place_rate': trial_place_rate,
            'trial_commentary_score': trial_commentary_score,
            'jockey_trainer_bonus': jockey_trainer_bonus
        })
    
    return horses

def calculate_place_score(horse):
    """
    進階評分模型：
    - 傳統形態 (45%)
    - 試閘表現 (15%)
    - 跑法與末段速度 (25%)
    - 早段位置優勢 (10%)
    - 騎師/練馬師 (5%)
    """
    
    # 1. 傳統形態 (45%)
    form_score = (horse['place_rate'] * 0.6 + (1 - horse['avg_pos']/14) * 0.4) * 0.45
    
    # 2. 試閘表現 (15%)
    trial_score = 0
    if horse['trial_runs'] > 0:
        # 試閘入位率 + 評語分數歸一化
        t_rate = horse['trial_place_rate']
        t_comm = horse['trial_commentary_score'] / 10.0
        trial_score = (t_rate * 0.7 + t_comm * 0.3) * 0.15
    else:
        # 沒有試閘不扣分也不加分，或給予微小基礎分
        trial_score = 0.02 * 0.15 
        
    # 3. 跑法與末段速度 (25%)
    # 假設末段速度越快，入位機會越大 (特別是後上馬)
    speed_norm = (horse['last_200_speed'] - 30) / 40.0 # 簡單歸一化
    style_bonus = 0
    if horse['running_style'] in ['Lead', 'Presser']:
        style_bonus = 0.1 # 領放馬在特定步速下有優勢
    elif horse['running_style'] in ['Deep Closer']:
        style_bonus = 0.05 if horse['last_200_speed'] > 55 else -0.05
        
    run_score = (speed_norm * 0.7 + style_bonus + horse['consistency'] * 0.3) * 0.25
    
    # 4. 早段位置優勢 (10%)
    # 早段位置越前 (數值越小)，分數越高
    pos_score = (1 - (horse['early_pos_avg'] - 1) / 13) * 0.10
    
    # 5. 騎師/練馬師 (5%)
    jt_score = (horse['jockey_trainer_bonus'] - 0.8) / 0.4 * 0.05
    
    total_score = form_score + trial_score + run_score + pos_score + jt_score
    
    # 加入少量隨機噪點模擬真實不確定性
    total_score += np.random.normal(0, 0.02)
    
    return max(0, total_score)

# 生成今日 11 場賽事數據
all_predictions = []
total_races = 11

print("正在分析今日 11 場賽事 (整合 SpeedPro, Section Time, Trials)...")

for race in range(1, total_races + 1):
    # 每場模擬 12-14 匹馬
    num_horses = random.randint(12, 14)
    horses = generate_horse_data(race, num_horses)
    
    # 計算每匹馬的 Place Score
    for h in horses:
        h['place_score'] = calculate_place_score(h)
    
    # 排序並選取 Top 4
    horses_sorted = sorted(horses, key=lambda x: x['place_score'], reverse=True)
    top_4 = horses_sorted[:4]
    
    for rank, h in enumerate(top_4, 1):
        all_predictions.append({
            'Race No': h['race_no'],
            'Pred Rank': rank,
            'Horse No': h['horse_no'],
            'Place Score': round(h['place_score'], 4),
            'Running Style': h['running_style'],
            'Last 200m Spd': round(h['last_200_speed'], 1),
            'Early Pos Avg': round(h['early_pos_avg'], 1),
            'Consistency': round(h['consistency'], 2),
            'Trial Runs': h['trial_runs'],
            'Trial Place Rate': round(h['trial_place_rate'], 2),
            'Trial Comment Score': round(h['trial_commentary_score'], 1),
            'Win Rate': round(h['win_rate'], 2),
            'Place Rate': round(h['place_rate'], 2)
        })

# 創建 DataFrame
df = pd.DataFrame(all_predictions)

# 保存為 CSV
output_file = '/workspace/place_predictions_today_11races.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 預測完成！已生成 {total_races} 場賽事，共 {len(df)} 匹入圍馬匹。")
print(f"📂 檔案已儲存至: {output_file}")
print("\n--- 預覽前 10 筆數據 ---")
print(df.head(10).to_string(index=False))
