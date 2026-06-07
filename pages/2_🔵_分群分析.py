import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="分群分析", page_icon="🔵", layout="wide")

st.markdown("""
<style>
.page-q { background:#f0f9f6; border-left:4px solid #1fa882; border-radius:0 8px 8px 0;
          padding:12px 16px; font-size:0.9rem; margin-bottom:20px; }
.cluster-card { border-radius:9px; padding:16px 18px; margin-bottom:6px; }
.cA { background:#eafaf3; border-left:4px solid #1fa882; }
.cB { background:#f3f4f6; border-left:4px solid #6b7280; }
.cC { background:#fff8e6; border-left:4px solid #f59e0b; }
.cluster-card h4 { margin:0 0 6px 0; font-size:0.9rem; }
.cluster-card ul { margin:0; padding-left:18px; font-size:0.82rem; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

st.title("🔵 K-Means 建築用電性格分群分析")
st.markdown('<div class="page-q">📌 本頁回答：<strong>11 棟建築的用電性格有哪些差異？哪些建築適合作為 DR 目標？K-Means 如何幫助我們做這個決定？</strong></div>', unsafe_allow_html=True)

@st.cache_data
def load_cluster():
    return pd.read_csv('./kmeans_cluster_results.csv')

@st.cache_data
def load_raw():
    df = pd.read_csv('./clean_vpp_data.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    return df

cluster_df = load_cluster()
df = load_raw()

COLOR_MAP = {'群 A：高彈性降載群':'#1fa882', '群 B：設備基載群':'#6b7280', '群 C：排程驅動群':'#f59e0b'}
DR_TARGETS = ['社科院大樓','管院一號館','霖澤館','新體育館','總圖書館']

# ── 方法說明 ───────────────────────────────────────────────
with st.expander("⚙️ 分群方法說明", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**特徵 1：基載率**")
        st.markdown("= P5% 用電 ÷ P95% 用電\n\n越低代表非尖峰時段用電驟降，彈性越大。")
    with c2:
        st.markdown("**特徵 2：相對冷氣敏感度**")
        st.markdown("= 氣溫回歸斜率 ÷ 尖峰用電 × 100\n\n標準化後可跨棟比較空調響應強度。")
    with c3:
        st.markdown("**為何不加第三特徵？**")
        st.markdown("嘗試加入「平假日差異」後 Silhouette 從 0.497 降至 0.204，且分群結果違反業務邏輯，故捨棄。改以氣泡大小作為輔助視覺維度。")
    st.metric("Silhouette Score（k=3）", "0.497", help="介於 -1 到 1，越高越好。11 個樣本點下屬中等品質，在業務邏輯驗證下可接受。")

st.markdown("---")

# ── 分群散佈圖 ─────────────────────────────────────────────
st.subheader("🗺️ 分群結果：11 棟建築全覽")

fig_scatter = go.Figure()
for grp_label, color in COLOR_MAP.items():
    sub = cluster_df[cluster_df['Cluster_Label'] == grp_label]
    for _, row in sub.iterrows():
        is_target = row['Building'] in DR_TARGETS
        # 氣泡大小 = 平假日差異（輔助維度）
        wd_we = row.get('Weekday_Weekend_Delta', 0.05)
        size = 14 + wd_we * 150
        fig_scatter.add_trace(go.Scatter(
            x=[row['Base_Load_Ratio']],
            y=[row['Relative_Cooling_Sensitivity']],
            mode='markers+text',
            name=grp_label,
            legendgroup=grp_label,
            showlegend=(list(sub['Building']).index(row['Building']) == 0),
            text=[row['Building']],
            textposition='top center',
            textfont=dict(size=10, color='#333'),
            marker=dict(
                color=color, size=size,
                symbol='star' if is_target else 'circle',
                line=dict(color='white', width=1.5),
                opacity=0.9
            ),
            hovertemplate=(
                f"<b>{row['Building']}</b><br>"
                f"基載率：{row['Base_Load_Ratio']:.3f}<br>"
                f"冷氣敏感度：{row['Relative_Cooling_Sensitivity']:.3f}<br>"
                f"DR 狀態：{row.get('DR_Status','—')}<extra></extra>"
            )
        ))

fig_scatter.add_vline(x=0.30, line_dash='dash', line_color='#94a3b8',
                      annotation_text='基載率 > 0.30（設備恆常運轉）',
                      annotation_position='bottom right',
                      annotation_font=dict(size=9, color='#94a3b8'))
fig_scatter.add_hline(y=2.00, line_dash='dash', line_color='#94a3b8',
                      annotation_text='冷氣敏感度 > 2.0（空調主導）',
                      annotation_position='top left',
                      annotation_font=dict(size=9, color='#94a3b8'))

fig_scatter.update_layout(
    xaxis_title='基載率（越低 = 用電彈性越大）',
    yaxis_title='相對冷氣敏感度（越高 = 氣溫影響越大）',
    template='plotly_white', height=520,
    legend=dict(orientation='h', y=1.06, title='分群（★ = DR 正式目標）'),
    xaxis=dict(range=[-0.02, 0.68]),
    yaxis=dict(range=[-0.1, 4.0])
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.caption("★ = 群 A 的 5 棟正式 DR 目標 · 氣泡大小 = 平假日用電差異比（輔助資訊，未納入分群計算）")

# ── 分群說明卡 ─────────────────────────────────────────────
cc1, cc2, cc3 = st.columns(3)
with cc1:
    st.markdown("""
<div class="cluster-card cA">
<h4>🟢 群 A：高彈性降載群（5 棟）</h4>
<ul>
  <li>社科院大樓 ★</li><li>管院一號館 ★</li><li>霖澤館 ★</li>
  <li>新體育館 ★</li><li>總圖書館 ★</li>
</ul>
<br><b>特徵</b>：低基載率 + 高冷氣敏感度<br>
<b>策略</b>：列為正式 DR 目標，空調調高 2°C 可有效降載。
</div>""", unsafe_allow_html=True)
with cc2:
    st.markdown("""
<div class="cluster-card cB">
<h4>⚪ 群 B：設備基載群（4 棟）</h4>
<ul>
  <li>凝態科學館</li><li>文學院</li>
  <li>生命科學館</li><li>管院二號館</li>
</ul>
<br><b>特徵</b>：高基載率（儀器設備 24H 運轉）<br>
<b>策略</b>：不列入降載目標，設為<b>研究對照組</b>。
</div>""", unsafe_allow_html=True)
with cc3:
    st.markdown("""
<div class="cluster-card cC">
<h4>🟡 群 C：排程驅動群（2 棟）</h4>
<ul>
  <li>共同教學館</li><li>新生大樓</li>
</ul>
<br><b>特徵</b>：用電受課表排程驅動，冷氣敏感度低<br>
<b>策略</b>：列入模擬作為<b>對照組</b>，驗證 K-Means 有效性。
<br><small>⚠️ 模型結果顯示降載效益確實有限，與分群結論一致。</small>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── 雷達圖：各群特徵對比 ──────────────────────────────────
st.subheader("🕸️ 各群用電特徵雷達圖")
st.caption("數值已 Min-Max 標準化；用電彈性 = 1 − 基載率")

radar_features = ['Base_Load_Ratio','Relative_Cooling_Sensitivity','Weekday_Weekend_Delta','Temp_Corr']
radar_labels   = ['用電彈性\n(=1−基載率)','冷氣敏感度','平假日差異','氣溫相關係數']
cluster_avg    = cluster_df.groupby('Cluster_Label')[radar_features].mean()
mins = cluster_df[radar_features].min()
maxs = cluster_df[radar_features].max()
cluster_scaled = (cluster_avg - mins) / (maxs - mins + 1e-9)
cluster_scaled['Base_Load_Ratio'] = 1 - cluster_scaled['Base_Load_Ratio']

N      = len(radar_features)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig_radar = go.Figure()
for grp, color in COLOR_MAP.items():
    if grp not in cluster_scaled.index:
        continue
    vals = cluster_scaled.loc[grp].tolist() + [cluster_scaled.loc[grp].tolist()[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals, theta=radar_labels + [radar_labels[0]],
        fill='toself', name=grp,
        line_color=color, fillcolor=color, opacity=0.25
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    showlegend=True, height=400, template='plotly_white',
    legend=dict(orientation='h', y=-0.15)
)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── 月均用電型態對比 ──────────────────────────────────────
st.subheader("📅 各群月均用電型態比較（% 年度最大值）")
st.caption("Y 軸正規化消除建築規模差異，聚焦於季節型態。實線 = DR 正式目標；紅底 = 夏季尖峰（6-9月）")

df2 = df.merge(cluster_df[['Building','Cluster_Label']], on='Building')
bldg_peak = df2.groupby('Building')['kW'].max()
monthly = df2.groupby(['Cluster_Label','Building','Month'])['kW'].mean().reset_index()
monthly['kW_pct'] = monthly.apply(lambda r: r['kW']/bldg_peak[r['Building']]*100, axis=1)
month_labels_short = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

tab_a, tab_b, tab_c = st.tabs(['群 A：高彈性降載群', '群 B：設備基載群', '群 C：排程驅動群'])
for tab, grp_label in zip([tab_a, tab_b, tab_c], COLOR_MAP.keys()):
    with tab:
        color = COLOR_MAP[grp_label]
        sub   = monthly[monthly['Cluster_Label']==grp_label]
        bldgs = cluster_df[cluster_df['Cluster_Label']==grp_label]['Building'].tolist()
        fig_m = go.Figure()
        for bldg in bldgs:
            bsub = sub[sub['Building']==bldg].sort_values('Month')
            is_target = bldg in DR_TARGETS
            fig_m.add_trace(go.Scatter(
                x=bsub['Month'], y=bsub['kW_pct'],
                name=f"{'★ ' if is_target else ''}{bldg}",
                mode='lines+markers',
                line=dict(width=2.5 if is_target else 1.5, dash='solid' if is_target else 'dash', color=color),
                marker=dict(size=6 if is_target else 4),
                opacity=0.95 if is_target else 0.55
            ))
        fig_m.add_vrect(x0=6, x1=9, fillcolor='rgba(239,68,68,0.07)', line_width=0,
                        annotation_text='夏季尖峰', annotation_position='top left',
                        annotation_font=dict(color='#ef4444', size=10))
        fig_m.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=month_labels_short),
            yaxis=dict(title='月均用電（% 年度最大值）', range=[0,100]),
            template='plotly_white', height=360,
            legend=dict(orientation='h', y=1.08)
        )
        st.plotly_chart(fig_m, use_container_width=True)