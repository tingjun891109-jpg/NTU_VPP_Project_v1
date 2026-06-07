"""
02_kmeans_clustering.py
========================
台大 VPP 專案 — K-Means 建築用電性格分群
升級版：新增 Elbow 曲線、雷達圖、月均用電曲線對比，並輸出分群結果 CSV

執行方式：python 02_kmeans_clustering.py
輸出：
  - kmeans_cluster_results.csv        ← 供 03 和 Streamlit 讀取
  - figures/02_elbow.png
  - figures/02_cluster_scatter.png
  - figures/02_radar.png
  - figures/02_monthly_profile.png

【特徵選擇說明】
本研究使用兩個特徵進行 K-Means 分群：
  1. 基載率（Base Load Ratio）：P5% ÷ P95%，衡量用電剛性
  2. 相對冷氣敏感度（Relative Cooling Sensitivity）：標準化空調負載斜率

曾評估加入第三個特徵「平假日用電差異比（Weekday-Weekend Delta）」，
但加入後 Silhouette Score 由 0.485 下降至 0.204，且社科院大樓與凝態科學館
被分至同一群（違反業務邏輯），故捨棄。平假日差異改以視覺標注方式呈現。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import platform

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
os.makedirs('./figures', exist_ok=True)

print('=' * 55)
print('  台大 VPP 專案 — K-Means 建築用電性格分群')
print('=' * 55)

# ==========================================
# 1. 載入資料
# ==========================================
print('\n【1】載入清洗後資料...')
df = pd.read_csv('./clean_vpp_data.csv')
df['Time'] = pd.to_datetime(df['Time'])
print(f'  資料筆數：{len(df):,}  ·  建築數量：{df["Building"].nunique()}')

# ==========================================
# 2. 特徵計算
# ==========================================
print('\n【2】計算分群特徵...')

features = []
for b in df['Building'].unique():
    b_data = df[df['Building'] == b]
    max_kw  = b_data['kW'].quantile(0.95)
    min_kw  = b_data['kW'].quantile(0.05)

    # 特徵 1：基載率（用電剛性指標）
    base_load_ratio = min_kw / max_kw if max_kw > 0 else 1

    # 特徵 2：相對冷氣敏感度（氣溫驅動強度，標準化後可跨棟比較）
    hot_data = b_data[b_data['Temperature'] > 20]
    if len(hot_data) > 10:
        slope, _ = np.polyfit(hot_data['Temperature'], hot_data['kW'], 1)
        rel_slope = max(0, (slope / max_kw) * 100)
    else:
        rel_slope = 0

    # 輔助資訊（不進入分群，供視覺化標注）
    wd = b_data[b_data['Is_Weekend'] == 0]['kW'].mean()
    we = b_data[b_data['Is_Weekend'] == 1]['kW'].mean()
    wd_we_ratio = (wd - we) / max_kw if max_kw > 0 else 0
    temp_corr   = b_data['kW'].corr(b_data['Temperature'])

    features.append({
        'Building':                    b,
        'Base_Load_Ratio':             round(base_load_ratio, 4),
        'Relative_Cooling_Sensitivity': round(rel_slope, 4),
        'Weekday_Weekend_Delta':        round(wd_we_ratio, 4),   # 輔助標注用
        'Temp_Corr':                    round(temp_corr, 4),     # 輔助標注用
        'Mean_kW':                      round(b_data['kW'].mean(), 2),
    })

feat_df = pd.DataFrame(features)

# ==========================================
# 3. 標準化 & K-Means（k=2~6 評估）
# ==========================================
print('\n【3】K-Means 分群（k=2~6 評估）...')

scaler      = StandardScaler()
scaled      = scaler.fit_transform(feat_df[['Base_Load_Ratio', 'Relative_Cooling_Sensitivity']])

elbow_k     = list(range(2, 7))
inertias    = []
sil_scores  = []

for k in elbow_k:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_k.fit(scaled)
    inertias.append(km_k.inertia_)
    sil_scores.append(silhouette_score(scaled, km_k.labels_))
    print(f'  k={k}:  Inertia={km_k.inertia_:.3f}   Silhouette={sil_scores[-1]:.4f}')

# 最終選擇 k=3（Silhouette 最高）
BEST_K   = 3
km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
feat_df['Cluster'] = km_final.fit_predict(scaled)
final_sil = silhouette_score(scaled, feat_df['Cluster'])

print(f'\n  ✅ 選定 k={BEST_K}，Silhouette Score = {final_sil:.4f}')

# 依業務邏輯命名分群
# Cluster 0：低基載率 + 高冷氣敏感度 → 高彈性降載群
# Cluster 1：高基載率 + 中冷氣敏感度 → 設備基載群
# Cluster 2：低基載率 + 低冷氣敏感度 → 排程驅動群
cluster_meta = {}
for c in range(BEST_K):
    sub = feat_df[feat_df['Cluster'] == c]
    cluster_meta[c] = {
        'mean_base':  sub['Base_Load_Ratio'].mean(),
        'mean_sens':  sub['Relative_Cooling_Sensitivity'].mean(),
        'buildings':  sub['Building'].tolist(),
    }

# 自動標籤（依重心排序）
sorted_by_sens = sorted(cluster_meta.items(), key=lambda x: x[1]['mean_sens'], reverse=True)
label_map  = {}
name_map   = {}
color_map  = {}
COLORS     = ['#1fa882', '#6b7280', '#f59e0b']

for rank, (cid, meta) in enumerate(sorted_by_sens):
    if rank == 0:
        label_map[cid] = '群 A：高彈性降載群'
        name_map[cid]  = 'A'
        color_map[cid] = '#1fa882'
    elif rank == 1:
        # 看基載率來區分
        if meta['mean_base'] > 0.30:
            label_map[cid] = '群 B：設備基載群'
            name_map[cid]  = 'B'
            color_map[cid] = '#6b7280'
        else:
            label_map[cid] = '群 C：排程驅動群'
            name_map[cid]  = 'C'
            color_map[cid] = '#f59e0b'
    else:
        if meta['mean_base'] > 0.30:
            label_map[cid] = '群 B：設備基載群'
            name_map[cid]  = 'B'
            color_map[cid] = '#6b7280'
        else:
            label_map[cid] = '群 C：排程驅動群'
            name_map[cid]  = 'C'
            color_map[cid] = '#f59e0b'

feat_df['Cluster_Label'] = feat_df['Cluster'].map(label_map)
feat_df['Cluster_Name']  = feat_df['Cluster'].map(name_map)
feat_df['Color']         = feat_df['Cluster'].map(color_map)

# DR 資格標注
# 群 A 全部可降載；群 C 列入模擬但效益預期低（排程驅動，非空調主導）
TARGET_BUILDINGS = ['共同教學館', '社科院大樓', '新生大樓', '管院一號館', '霖澤館']

def dr_status(row):
    if row['Cluster_Name'] == 'A':
        return '✅ DR 目標'
    elif row['Building'] in TARGET_BUILDINGS:
        return '⚠️ 列入模擬，效益預期低'
    else:
        return '❌ 非 DR 目標'

feat_df['DR_Status'] = feat_df.apply(dr_status, axis=1)

print('\n=== 最終分群結果 ===')
print(feat_df[['Building', 'Cluster_Label', 'DR_Status',
               'Base_Load_Ratio', 'Relative_Cooling_Sensitivity']].to_string(index=False))

# ==========================================
# 4. 存檔
# ==========================================
output_cols = ['Building', 'Cluster', 'Cluster_Label', 'Cluster_Name',
               'DR_Status', 'Base_Load_Ratio', 'Relative_Cooling_Sensitivity',
               'Weekday_Weekend_Delta', 'Temp_Corr', 'Mean_kW']
feat_df[output_cols].to_csv('./kmeans_cluster_results.csv', index=False, encoding='utf-8-sig')
print('\n✅ 分群結果已儲存：kmeans_cluster_results.csv')

# ==========================================
# 5. 視覺化
# ==========================================

# ── 圖 1：Elbow 曲線（k 選擇依據）─────────────────────────
print('\n【5】產出視覺化圖表...')
print('  繪製圖 1：Elbow + Silhouette 雙軸曲線...')

fig1, ax1a = plt.subplots(figsize=(8, 4.5))
ax1b = ax1a.twinx()

color_elbow = '#3b82f6'
color_sil   = '#1fa882'

ax1a.plot(elbow_k, inertias, 'o-', color=color_elbow, linewidth=2.5,
          markersize=8, label='Inertia（組內距離平方和）')
ax1a.fill_between(elbow_k, inertias, alpha=0.08, color=color_elbow)
ax1a.set_xlabel('群數 k', fontsize=12)
ax1a.set_ylabel('Inertia', fontsize=12, color=color_elbow)
ax1a.tick_params(axis='y', labelcolor=color_elbow)

ax1b.plot(elbow_k, sil_scores, 's--', color=color_sil, linewidth=2.5,
          markersize=8, label='Silhouette Score')
ax1b.set_ylabel('Silhouette Score', fontsize=12, color=color_sil)
ax1b.tick_params(axis='y', labelcolor=color_sil)

# 標注最佳 k
best_sil_idx = sil_scores.index(max(sil_scores))
ax1b.annotate(
    f'最佳 k={elbow_k[best_sil_idx]}\nSilhouette={max(sil_scores):.4f}',
    xy=(elbow_k[best_sil_idx], max(sil_scores)),
    xytext=(elbow_k[best_sil_idx] + 0.3, max(sil_scores) - 0.04),
    fontsize=10, color=color_sil,
    arrowprops=dict(arrowstyle='->', color=color_sil, lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e0f5ee', edgecolor=color_sil)
)

ax1a.set_title('圖 1：Elbow 曲線 — K-Means 最適群數選擇\n（Inertia 下降趨緩 + Silhouette 最高點 → k=3）',
               fontsize=12, pad=10)
ax1a.set_xticks(elbow_k)

lines1, labels1 = ax1a.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1a.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax1a.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('./figures/02_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/02_elbow.png')

# ── 圖 2：分群散佈圖（平假日差異 = 點大小標注）─────────────
print('  繪製圖 2：分群散佈圖...')

fig2, ax2 = plt.subplots(figsize=(11, 7))

# 平假日差異標準化為點大小（50~300）
delta_vals  = feat_df['Weekday_Weekend_Delta'].values
delta_norm  = (delta_vals - delta_vals.min()) / (delta_vals.max() - delta_vals.min() + 1e-9)
bubble_size = 80 + delta_norm * 280

for _, row in feat_df.iterrows():
    idx  = feat_df.index.get_loc(_)
    ax2.scatter(
        row['Base_Load_Ratio'], row['Relative_Cooling_Sensitivity'],
        s=bubble_size[idx],
        color=row['Color'],
        alpha=0.85,
        edgecolors='white', linewidths=1.5,
        zorder=3
    )
    # 建築名稱標注
    offset_x = 0.012
    offset_y = 0.06 if row['Building'] not in ['霖澤館', '管院二號館'] else -0.12
    ax2.text(
        row['Base_Load_Ratio'] + offset_x,
        row['Relative_Cooling_Sensitivity'] + offset_y,
        row['Building'],
        fontsize=9.5, va='center', zorder=4,
        color='#333'
    )

# 參考線
ax2.axvline(x=0.30, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.6)
ax2.axhline(y=2.00, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.6)
ax2.text(0.31, 0.15, '基載率 > 0.30\n（設備恆常運轉）', fontsize=8.5, color='#94a3b8')
ax2.text(0.01, 2.05, '冷氣敏感度 > 2.0\n（空調主導）', fontsize=8.5, color='#94a3b8')

# 群心標示
for c in range(BEST_K):
    center = scaler.inverse_transform([km_final.cluster_centers_[c]])[0]
    ax2.scatter(center[0], center[1], s=200, marker='+',
                color=color_map[c], linewidths=2.5, zorder=5)

# 圖例（群）
legend_patches = [
    mpatches.Patch(color=color_map[c], label=label_map[c])
    for c in range(BEST_K)
]
# 氣泡大小圖例（平假日差異）
for size, label in [(80, '低'), (200, '中'), (360, '高')]:
    legend_patches.append(
        plt.scatter([], [], s=size, color='#cbd5e1',
                    edgecolors='#888', label=f'平假日差異：{label}')
    )

ax2.legend(handles=legend_patches, loc='upper right', fontsize=9,
           title='分群  ·  氣泡大小 = 平假日用電差異比', title_fontsize=9)

ax2.set_xlabel('基載率（越低 = 用電彈性越大）', fontsize=12)
ax2.set_ylabel('相對冷氣敏感度（越高 = 氣溫影響越大）', fontsize=12)
ax2.set_title('圖 2：K-Means 分群散佈圖（k=3）\n氣泡大小 = 平假日用電差異比（輔助資訊，未納入分群計算）',
              fontsize=12, pad=12)
ax2.grid(True, linestyle='--', alpha=0.35)
ax2.set_xlim(-0.02, 0.70)
ax2.set_ylim(-0.1, 4.2)
plt.tight_layout()
plt.savefig('./figures/02_cluster_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/02_cluster_scatter.png')

# ── 圖 3：各群重心雷達圖（三群特徵對比）──────────────────
print('  繪製圖 3：分群雷達圖...')

# 準備雷達圖資料：4 個維度，Min-Max 標準化
from matplotlib.patches import FancyArrowPatch

radar_features = ['Base_Load_Ratio', 'Relative_Cooling_Sensitivity',
                  'Weekday_Weekend_Delta', 'Temp_Corr']
radar_labels   = ['基載率\n（低=彈性大）', '冷氣敏感度\n（高=空調主導）',
                  '平假日差異\n（高=辦公屬性）', '氣溫相關係數\n（高=降載潛力大）']

# 計算各群平均值並 Min-Max 標準化
cluster_avg = feat_df.groupby('Cluster')[radar_features].mean()
mins = feat_df[radar_features].min()
maxs = feat_df[radar_features].max()
cluster_scaled = (cluster_avg - mins) / (maxs - mins + 1e-9)

# 基載率反向（越低越好，顯示時反轉讓「大 = 好」）
cluster_scaled['Base_Load_Ratio'] = 1 - cluster_scaled['Base_Load_Ratio']
radar_labels[0] = '用電彈性\n（=1-基載率）'

N        = len(radar_features)
angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles  += angles[:1]

fig3, axes3 = plt.subplots(1, BEST_K, figsize=(15, 5),
                            subplot_kw=dict(polar=True))

for ax, c in zip(axes3, range(BEST_K)):
    values  = cluster_scaled.loc[c].tolist() + [cluster_scaled.loc[c].tolist()[0]]
    color   = color_map[c]
    label   = label_map[c]
    bldgs   = feat_df[feat_df['Cluster'] == c]['Building'].tolist()

    ax.plot(angles, values, color=color, linewidth=2.5, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=7, color='gray')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # 標題含建築名單
    bldg_str = '\n'.join(bldgs)
    ax.set_title(f'{label}\n\n{bldg_str}', fontsize=10, pad=20,
                 color=color, fontweight='bold')

fig3.suptitle('圖 3：各分群用電特徵雷達圖\n（數值已 Min-Max 標準化；用電彈性 = 1 - 基載率）',
              fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('./figures/02_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/02_radar.png')

# ── 圖 4：月均用電曲線（各棟正規化為年度最大值百分比）────
print('  繪製圖 4：月均用電曲線對比（正規化版）...')

# 【修正說明】
# 原始版用絕對 kW 繪圖，導致群內規模差異大的建築（例如群 A 的總圖書館 620 kW
# 與霖澤館 40 kW）在同一張圖中，小建築的線貼近 0 看不出季節型態。
# 解法：改用「各棟年度最大值的百分比」(% of annual peak)，消除規模差異，
# 讓不同大小的建築可以在同一 Y 軸上比較「季節性用電型態」。

df_merged  = df.merge(feat_df[['Building', 'Cluster', 'Cluster_Label']], on='Building')

# 各棟年度最大值
bldg_peak  = df_merged.groupby('Building')['kW'].max()

# 月均後正規化
monthly_df = df_merged.groupby(['Cluster_Label', 'Building', 'Month'])['kW'].mean().reset_index()
monthly_df['kW_pct'] = monthly_df.apply(
    lambda r: r['kW'] / bldg_peak[r['Building']] * 100, axis=1
)

month_labels = ['1月','2月','3月','4月','5月','6月',
                '7月','8月','9月','10月','11月','12月']

# 各群對應的洞察文字標注
GROUP_INSIGHTS = {
    # 格式：(文字, 箭頭x, 箭頭y, 文字框x, 文字框y, 字色, 背景色, 框色)
    # 群 A：夏季 40-54%，冬季 18-27%，箭頭指向 9 月社科院高點
    '群 A：高彈性降載群': ('夏季明顯上揚\n（空調負載主導）',   9, 46,  5, 68, '#0f7c6e', '#e0f5ee', '#1fa882'),
    # 群 B：凝態 56-78%（高載），其餘三棟 36-63%（中等），箭頭指向凝態 9 月高點
    '群 B：設備基載群':   ('凝態高載（恆常運轉）\n其餘三棟全年平穩', 9, 77,  4, 55, '#374151', '#f3f4f6', '#6b7280'),
    # 群 C：夏季 25-43%，幾乎無季節差異，箭頭指向 8 月共同教學館
    '群 C：排程驅動群':   ('夏季無明顯上揚\n（用電受課表驅動）',  8, 40,  3, 68, '#92400e', '#fffbeb', '#f59e0b'),
}

fig4, axes4 = plt.subplots(1, BEST_K, figsize=(17, 5.5), sharey=True)

for ax, c in zip(axes4, range(BEST_K)):
    grp_label = label_map[c]
    color     = color_map[c]
    bldgs_in  = feat_df[feat_df['Cluster'] == c]['Building'].tolist()
    sub       = monthly_df[monthly_df['Cluster_Label'] == grp_label]

    for bldg in bldgs_in:
        bsub      = sub[sub['Building'] == bldg].sort_values('Month')
        is_target = bldg in TARGET_BUILDINGS
        ax.plot(
            bsub['Month'], bsub['kW_pct'],
            linewidth=2.5 if is_target else 1.5,
            alpha=0.95 if is_target else 0.55,
            linestyle='-' if is_target else '--',
            color=color,
            marker='o', markersize=5 if is_target else 3,
            label=f'{"★ " if is_target else ""}{bldg}'
        )

    # 夏季陰影
    ax.axvspan(6, 9, alpha=0.07, color='#ef4444')
    ax.text(7.5, 94, '夏季\n尖峰', fontsize=8, color='#ef4444',
            ha='center', va='top', alpha=0.7)

    # 洞察標注
    if grp_label in GROUP_INSIGHTS:
        txt, ax_x, ax_y, tx_x, tx_y, fc, bgc, ec = GROUP_INSIGHTS[grp_label]
        ax.annotate(
            txt, xy=(ax_x, ax_y), xytext=(tx_x, tx_y),
            fontsize=8, color=fc,
            arrowprops=dict(arrowstyle='->', color=ec, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bgc, edgecolor=ec, alpha=0.9)
        )

    ax.set_title(grp_label, fontsize=11, color=color, fontweight='bold', pad=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=30, fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('月均用電（% 年度最大值）', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper left',
              title='實線=03模擬目標' if grp_label != '群 B：設備基載群' else '',
              title_fontsize=7.5)
    ax.grid(True, linestyle='--', alpha=0.4)

fig4.suptitle(
    '圖 4：各群建築月均用電型態比較\n'
    '（Y 軸 = 各棟年度最大值的百分比，消除建築規模差異；'
    '實線 = 03 模擬目標；紅底 = 夏季尖峰）',
    fontsize=12, y=1.03
)
plt.tight_layout()
plt.savefig('./figures/02_monthly_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✅ figures/02_monthly_profile.png')

# ==========================================
# 6. 最終摘要
# ==========================================
print()
print('=' * 55)
print('  分群摘要')
print('=' * 55)
for c in range(BEST_K):
    sub = feat_df[feat_df['Cluster'] == c]
    print(f'\n  {label_map[c]}（{len(sub)} 棟）')
    print(f'  建築：{", ".join(sub["Building"].tolist())}')
    print(f'  平均基載率：{sub["Base_Load_Ratio"].mean():.3f}')
    print(f'  平均冷氣敏感度：{sub["Relative_Cooling_Sensitivity"].mean():.3f}')
    dr_bldgs = sub[sub['DR_Status'] != '❌ 非 DR 目標']['Building'].tolist()
    if dr_bldgs:
        print(f'  DR 相關：{", ".join(dr_bldgs)}')

print(f'\n  Silhouette Score（k=3）：{final_sil:.4f}')
print(f'\n  ✅ 所有圖表已輸出至 ./figures/')
print(f'  ✅ 分群結果已儲存：kmeans_cluster_results.csv')
print('=' * 55)