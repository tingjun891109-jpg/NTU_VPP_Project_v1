import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# Check OS and set font
if platform.system() == 'Windows':
    font_name = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':
    font_name = 'PingFang TC'
else:
    font_name = 'sans-serif'

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=font_name)

# 1. Load the clean CSV file
print("Loading clean data...")
df = pd.read_csv('./clean_vpp_data.csv')
df['Time'] = pd.to_datetime(df['Time'])

# 2. Extract features for each building (嚴謹修正版)
features = []
buildings = df['Building'].unique()

print("Calculating features...")
for b in buildings:
    b_data = df[df['Building'] == b]
    
    # 特徵 1：基載率 (維持原邏輯)
    min_kw = b_data['kW'].quantile(0.05)
    max_kw = b_data['kW'].quantile(0.95)
    base_load_ratio = min_kw / max_kw if max_kw > 0 else 1
    
    # 特徵 2：相對冷氣敏感度 (Relative Cooling Sensitivity)
    hot_data = b_data[b_data['Temperature'] > 20]
    
    if len(hot_data) > 10:
        # 先算出絕對斜率 (kW / °C)
        slope, _ = np.polyfit(hot_data['Temperature'], hot_data['kW'], 1)
        # 移除 max(0) 限制，觀察是否有異常的負相關
        
        # 【關鍵修正】：將絕對 kW 轉化為「佔該建築尖峰用電的百分比」
        # 這樣才能在同一基準線上比較大樓與小樓的空調彈性
        relative_slope = (slope / max_kw) * 100 if max_kw > 0 else 0
        
        # 若為負值（氣溫越高用電越低，不符合物理邏輯），強制歸零
        relative_slope = max(0, relative_slope)
    else:
        relative_slope = 0
        
    features.append({
        'Building': b,
        'Base_Load_Ratio': base_load_ratio,
        'Relative_Cooling_Sensitivity': relative_slope # 變數名稱更新
    })

feature_df = pd.DataFrame(features)

# 3. K-Means Clustering
# Standardize data so both features have equal weight
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df[['Base_Load_Ratio', 'Relative_Cooling_Sensitivity']])

# Group into 3 categories
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
feature_df['Cluster'] = kmeans.fit_predict(scaled_features)

print("\nClustering Results:")
print(feature_df.sort_values('Cluster'))

# 4. Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=feature_df, 
    x='Base_Load_Ratio', 
    y='Relative_Cooling_Sensitivity', 
    hue='Cluster', 
    palette='Set2', 
    s=150,
    edgecolor='black'
)

# Add building names next to the dots
for i in range(feature_df.shape[0]):
    plt.text(
        feature_df['Base_Load_Ratio'][i] + 0.01, 
        feature_df['Relative_Cooling_Sensitivity'][i], 
        feature_df['Building'][i], 
        fontsize=10
    )

plt.title('台大校園建築 VPP 降載潛力分群 (K-Means)', fontsize=16, pad=15)
plt.xlabel('基載率 (越低代表卸載彈性越大)', fontsize=12)
plt.ylabel('冷氣敏感度 (kW / °C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Move legend outside
plt.legend(title='分群結果', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()