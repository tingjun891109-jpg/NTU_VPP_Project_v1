"""
01_data_cleaning.py
====================
台大 VPP 專案 — 資料清洗與 ETL 流程
升級版：新增資料品質報告、清洗前後對比圖、缺失值熱力圖

執行方式：python 01_data_cleaning.py
輸出：
  - clean_vpp_data.csv        ← Streamlit 儀表板使用
  - figures/01_missing_heatmap.png
  - figures/01_before_after.png
  - figures/01_temp_power_corr.png
"""

import pandas as pd
import numpy as np
import glob
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ==========================================
# 0. 環境設定
# ==========================================
if platform.system() == 'Windows':
    font_name = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':
    font_name = 'PingFang TC'
else:
    font_name = 'Noto Sans CJK TC' if os.path.exists('/usr/share/fonts/opentype/noto') else 'sans-serif'

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font=font_name)

# 建立 figures 輸出資料夾
os.makedirs('./figures', exist_ok=True)

folder_path   = './raw_data/'
excel_files   = sorted(glob.glob(os.path.join(folder_path, '*.xlsx')))
csv_files     = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

print('=' * 55)
print('  台大 VPP 專案 — 資料 ETL 與品質報告')
print('=' * 55)
print(f'  找到 Excel 電錶檔案：{len(excel_files)} 個')
print(f'  找到 CODiS 氣象檔案：{len(csv_files)} 個')
print()

# ==========================================
# 階段一：讀取 11 棟建築電力資料
# ==========================================
print('【階段一】讀取電錶資料...')

BUILDING_NAME_MAP = {
    '共同': '共同教學館',
    '凝態': '凝態科學館',
    '文學': '文學院',
    '新生': '新生大樓',
    '新體': '新體育館',
    '生命': '生命科學館',
    '社科': '社科院大樓',
    '管院一': '管院一號館',
    '管院二': '管院二號館',
    '總圖': '總圖書館',
    '霖澤': '霖澤館',
}

def normalize_building_name(filename):
    """將各種檔名格式統一對應到標準建築名稱"""
    name = os.path.basename(filename).replace('.xlsx', '').lstrip('_').lstrip('~$')
    for key, standard in BUILDING_NAME_MAP.items():
        if key in name:
            return standard
    return name  # 找不到對應則保留原始名稱

df_list        = []
raw_stats      = []   # 記錄每棟原始資料統計，供品質報告使用

for file in excel_files:
    if os.path.basename(file).startswith('~$'):
        continue
    try:
        temp_df = pd.read_excel(file, skiprows=1, header=None)
        temp_df = temp_df.iloc[:, [0, 1]].copy()
        temp_df.columns = ['Time', 'kW']

        building_name = normalize_building_name(file)
        temp_df['Building'] = building_name
        temp_df['kW'] = pd.to_numeric(temp_df['kW'], errors='coerce')
        temp_df['Time'] = pd.to_datetime(temp_df['Time'], errors='coerce')
        temp_df = temp_df.dropna(subset=['Time'])

        raw_stats.append({
            'Building':      building_name,
            '原始筆數':       len(temp_df),
            '原始缺失_kW':    temp_df['kW'].isna().sum(),
            '原始負值_kW':    (temp_df['kW'] < 0).sum(),
        })

        df_list.append(temp_df)
        print(f'  ✅ {building_name:<10} — {len(temp_df):>6,} 筆')

    except Exception as e:
        print(f'  ❌ 讀取失敗 {os.path.basename(file)}: {e}')

if not df_list:
    raise RuntimeError('找不到任何可讀取的 xlsx 檔案，請確認 raw_data/ 路徑。')

raw_data = pd.concat(df_list, ignore_index=True)
raw_data = raw_data.sort_values(by=['Building', 'Time']).reset_index(drop=True)
# 保留原始資料副本以供後續視覺化對比
raw_data_backup = raw_data.copy()

print(f'\n  合計：{len(raw_data):,} 筆原始電力資料\n')

# ==========================================
# 階段二：讀取 CODiS 氣象資料
# ==========================================
print('【階段二】讀取氣象資料（CODiS MH 格式）...')

# CODiS MH 格式欄位說明（# 開頭行）：
# stno, yyyymmddhh, PS01, PS02, TX01(氣溫), TD01, RH01(濕度), ...
# 我們只取 TX01（乾球溫度，第 5 欄，index 4）

weather_records = []
files_read      = 0

for file in csv_files:
    if 'DataState' in os.path.basename(file):
        continue
    try:
        with open(file, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('466920'):
                    continue
                cols = [c.strip() for c in line.split(',')]
                if len(cols) < 5:
                    continue
                time_str = cols[1].strip()
                temp_str = cols[4].strip()   # TX01 乾球溫度
                weather_records.append([time_str, temp_str])
        files_read += 1
    except Exception as e:
        print(f'  ⚠️  讀取氣象檔失敗 {os.path.basename(file)}: {e}')

print(f'  讀取氣象檔：{files_read} 個，共 {len(weather_records)} 筆時間點')

# 解析時間（CODiS 使用 24 時制：2025010124 = 2025/01/02 00:00）
def parse_codis_time(t_str):
    t_str = str(t_str).strip()
    if len(t_str) == 10 and t_str[-2:] == '24':
        base = pd.to_datetime(t_str[:-2] + '00', format='%Y%m%d%H')
        return base + pd.Timedelta(days=1)
    return pd.to_datetime(t_str, format='%Y%m%d%H', errors='coerce')

weather_df = pd.DataFrame(weather_records, columns=['TimeStr', 'Temperature'])
weather_df['Time']        = weather_df['TimeStr'].apply(parse_codis_time)
weather_df['Temperature'] = pd.to_numeric(weather_df['Temperature'], errors='coerce')
weather_df = (weather_df
              .dropna(subset=['Time'])
              .drop_duplicates(subset=['Time'])
              .sort_values('Time')
              [['Time', 'Temperature']]
              .reset_index(drop=True))

print(f'  氣象資料時間範圍：{weather_df["Time"].min()} ～ {weather_df["Time"].max()}')
print(f'  有效氣溫筆數：{weather_df["Temperature"].notna().sum():,}\n')

# ==========================================
# 階段三：合併 & 特徵工程
# ==========================================
print('【階段三】合併資料 & 特徵工程...')

merged = pd.merge(raw_data, weather_df, on='Time', how='left')

# 時間特徵
merged['Hour']       = merged['Time'].dt.hour
merged['Month']      = merged['Time'].dt.month
merged['Is_Weekend'] = merged['Time'].dt.dayofweek.isin([5, 6]).astype(int)

print(f'  合併後筆數：{len(merged):,}')
print(f'  氣溫缺失率：{merged["Temperature"].isna().mean()*100:.1f}%\n')

# ==========================================
# 階段四：資料清洗
# ==========================================
print('【階段四】資料清洗...')

clean = merged.copy()

# 4-1：修正異常氣溫（> 45°C 或 ≤ 0°C 視為感應器異常）
temp_anomaly = ((clean['Temperature'] <= 0) | (clean['Temperature'] >= 45))
print(f'  氣溫異常筆數（感應器錯誤）：{temp_anomaly.sum()}')
clean.loc[temp_anomaly, 'Temperature'] = pd.NA
clean['Temperature'] = (clean.groupby('Building')['Temperature']
                        .transform(lambda x: x.interpolate(method='linear').ffill().bfill()))

# 4-2：移除零值 / 負值電力（儀器斷線或資料錯誤）
zero_neg = (clean['kW'] <= 0)
print(f'  零值 / 負值電力筆數：{zero_neg.sum()}')
clean = clean[~zero_neg]

# 4-3：IQR 法去除尖峰異常值（每棟獨立計算）
before_iqr = len(clean)

def remove_iqr_spikes(group):
    q1  = group['kW'].quantile(0.25)
    q3  = group['kW'].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    removed = (group['kW'] > upper).sum()
    return group[group['kW'] <= upper], removed

clean_list      = []
iqr_removed_map = {}
for bldg, grp in clean.groupby('Building'):
    cleaned_grp, n_removed = remove_iqr_spikes(grp)
    clean_list.append(cleaned_grp)
    iqr_removed_map[bldg] = n_removed

clean      = pd.concat(clean_list).sort_values(['Building', 'Time']).reset_index(drop=True)
after_iqr  = len(clean)
print(f'  IQR 異常移除：{before_iqr - after_iqr} 筆（各棟獨立計算 Q3 + 1.5×IQR）')

# ==========================================
# 階段五：資料品質報告（文字版）
# ==========================================
print()
print('=' * 55)
print('  資料品質報告（清洗後）')
print('=' * 55)
print(f'  {"建築":<12} {"清洗後筆數":>8} {"缺失率%":>8} {"IQR移除":>8} {"kW均值":>8} {"kW最大":>8}')
print('  ' + '-' * 53)

for bldg in sorted(clean['Building'].unique()):
    b   = clean[clean['Building'] == bldg]
    raw = next((r for r in raw_stats if r['Building'] == bldg), {})
    print(f'  {bldg:<12} {len(b):>8,} {b["kW"].isna().mean()*100:>7.1f}% '
          f'{iqr_removed_map.get(bldg, 0):>8} '
          f'{b["kW"].mean():>8.1f} {b["kW"].max():>8.1f}')

total_removed = len(merged) - len(clean)
print(f'\n  原始資料：{len(merged):,} 筆 → 清洗後：{len(clean):,} 筆')
print(f'  移除比例：{total_removed/len(merged)*100:.1f}%（含零值、IQR 異常）')
print()

# ==========================================
# 階段六：存檔
# ==========================================
output_path = './clean_vpp_data.csv'
clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f'✅ 乾淨資料已儲存：{output_path}  ({len(clean):,} 筆)\n')

# ==========================================
# 階段七：視覺化輸出
# ==========================================
print('【階段七】產出視覺化圖表...')

# ── 圖 1：缺失值熱力圖（各建築 × 月份）────────────────────
print('  繪製圖 1：缺失值分布熱力圖...')

# 建立完整時間軸（每棟建築每小時都應有一筆）
expected_hours = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='h')
buildings      = sorted(clean['Building'].unique())

# 計算每棟建築每月的缺失率
missing_matrix = {}
for bldg in buildings:
    bdata = clean[clean['Building'] == bldg].set_index('Time')['kW']
    bdata = bdata.reindex(expected_hours)
    monthly_missing = bdata.resample('ME').apply(lambda x: x.isna().mean() * 100)
    missing_matrix[bldg] = monthly_missing

missing_df = pd.DataFrame(missing_matrix).T
missing_df.columns = [f'{i+1}月' for i in range(len(missing_df.columns))]

fig1, ax1 = plt.subplots(figsize=(13, 5))
sns.heatmap(
    missing_df, annot=True, fmt='.1f', cmap='YlOrRd',
    linewidths=0.4, ax=ax1,
    cbar_kws={'label': '缺失率 (%)'},
    vmin=0, vmax=10
)
ax1.set_title('圖 1：各建築逐月用電資料缺失率 (%)\n（清洗後，含插補）', fontsize=13, pad=12)
ax1.set_xlabel('月份', fontsize=11)
ax1.set_ylabel('建築', fontsize=11)
ax1.tick_params(axis='x', rotation=0)
ax1.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('./figures/01_missing_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/01_missing_heatmap.png')

# ── 圖 2：清洗前後用電分布對比（各棟 Box Plot）────────────
print('  繪製圖 2：清洗前後用電分布對比...')

fig2, axes = plt.subplots(1, 2, figsize=(15, 6))

# 只取已成功讀取的建築做對比
common_bldgs = sorted(set(raw_data_backup['Building'].unique()) &
                      set(clean['Building'].unique()))

# 清洗前
raw_plot = raw_data_backup[raw_data_backup['Building'].isin(common_bldgs)].copy()
raw_plot = raw_plot.sort_values('Building')
bp1 = axes[0].boxplot(
    [raw_plot[raw_plot['Building'] == b]['kW'].dropna().values for b in common_bldgs],
    labels=common_bldgs, patch_artist=True,
    flierprops=dict(marker='.', markersize=2, alpha=0.3, color='#e74c3c'),
    medianprops=dict(color='#e74c3c', linewidth=1.5),
)
for patch in bp1['boxes']:
    patch.set_facecolor('#fde8e8')
    patch.set_alpha(0.8)
axes[0].set_title('清洗前：原始用電分布', fontsize=12, pad=8)
axes[0].set_ylabel('用電功率 (kW)', fontsize=11)
axes[0].tick_params(axis='x', rotation=30, labelsize=9)
axes[0].set_xlabel('')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# 清洗後
clean_plot = clean[clean['Building'].isin(common_bldgs)].copy()
clean_plot = clean_plot.sort_values('Building')
bp2 = axes[1].boxplot(
    [clean_plot[clean_plot['Building'] == b]['kW'].dropna().values for b in common_bldgs],
    labels=common_bldgs, patch_artist=True,
    flierprops=dict(marker='.', markersize=2, alpha=0.3, color='#1fa882'),
    medianprops=dict(color='#1fa882', linewidth=1.5),
)
for patch in bp2['boxes']:
    patch.set_facecolor('#e0f5ee')
    patch.set_alpha(0.8)
axes[1].set_title('清洗後：IQR 異常移除（Q3 + 1.5×IQR）', fontsize=12, pad=8)
axes[1].set_ylabel('用電功率 (kW)', fontsize=11)
axes[1].tick_params(axis='x', rotation=30, labelsize=9)
axes[1].set_xlabel('')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

fig2.suptitle('圖 2：清洗前後各建築用電功率分布對比', fontsize=14, y=1.01, fontweight='bold')
plt.tight_layout()
plt.savefig('./figures/01_before_after.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/01_before_after.png')

# ── 圖 3：氣溫 × 用電散佈圖（6 棟代表性建築）────────────
print('  繪製圖 3：氣溫 × 用電相關性散佈圖...')

# 挑選 6 棟代表不同特性的建築
SHOWCASE = ['社科院大樓', '管院一號館', '凝態科學館', '霖澤館', '共同教學館', '總圖書館']
SHOWCASE = [b for b in SHOWCASE if b in clean['Building'].unique()]

# 補足到 6 棟
for b in sorted(clean['Building'].unique()):
    if b not in SHOWCASE and len(SHOWCASE) < 6:
        SHOWCASE.append(b)

colors = ['#1fa882', '#3b82f6', '#e11d48', '#f59e0b', '#8b5cf6', '#64748b']
fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
axes3_flat = axes3.flatten()

for i, bldg in enumerate(SHOWCASE[:6]):
    ax = axes3_flat[i]
    bdata = clean[clean['Building'] == bldg].sample(min(1500, len(clean[clean['Building'] == bldg])), random_state=42)
    r = bdata['kW'].corr(bdata['Temperature'])

    ax.scatter(bdata['Temperature'], bdata['kW'],
               alpha=0.25, s=6, color=colors[i], rasterized=True)

    # 回歸線
    hot = bdata[bdata['Temperature'] > 20]
    if len(hot) > 20:
        z = np.polyfit(hot['Temperature'], hot['kW'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(hot['Temperature'].min(), hot['Temperature'].max(), 100)
        ax.plot(x_line, p(x_line), color=colors[i], linewidth=2, label=f'回歸線（r={r:.2f}）')

    ax.set_title(f'{bldg}', fontsize=11, fontweight='bold')
    ax.set_xlabel('氣溫 (°C)', fontsize=9)
    ax.set_ylabel('用電功率 (kW)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=8)

fig3.suptitle('圖 3：氣溫 vs. 用電功率散佈圖（含回歸線，r = 皮爾森相關係數）',
              fontsize=13, y=1.01, fontweight='bold')
plt.tight_layout()
plt.savefig('./figures/01_temp_power_corr.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/01_temp_power_corr.png')

# ── 圖 4：每月平均氣溫趨勢（確認氣象資料品質）────────────
print('  繪製圖 4：逐月氣溫趨勢...')

monthly_temp = (clean.groupby('Month')['Temperature'].mean().reset_index())
monthly_temp.columns = ['Month', 'AvgTemp']

fig4, ax4 = plt.subplots(figsize=(9, 4))
ax4.bar(monthly_temp['Month'], monthly_temp['AvgTemp'],
        color=['#3b82f6' if t < 26 else '#f59e0b' if t < 30 else '#ef4444'
               for t in monthly_temp['AvgTemp']],
        edgecolor='white', linewidth=0.5)
ax4.plot(monthly_temp['Month'], monthly_temp['AvgTemp'],
         'o-', color='#1e3a5f', linewidth=1.8, markersize=6, zorder=5)

for _, row in monthly_temp.iterrows():
    ax4.text(row['Month'], row['AvgTemp'] + 0.3, f"{row['AvgTemp']:.1f}°C",
             ha='center', va='bottom', fontsize=9, color='#333')

ax4.axhline(y=28, color='#ef4444', linestyle='--', linewidth=1, alpha=0.6, label='28°C（高溫警戒）')
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels([f'{m}月' for m in range(1, 13)], fontsize=9)
ax4.set_ylabel('月平均氣溫 (°C)', fontsize=10)
ax4.set_title('圖 4：2025 年逐月平均氣溫（台大氣象站 466920）', fontsize=12, pad=10)
ax4.legend(fontsize=9)
ax4.set_ylim(0, max(monthly_temp['AvgTemp']) + 3)
ax4.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('./figures/01_monthly_temp.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/01_monthly_temp.png')

print()
print('=' * 55)
print('  ✅ 所有圖表已輸出至 ./figures/')
print('  ✅ ETL 流程完成！')
print('=' * 55)