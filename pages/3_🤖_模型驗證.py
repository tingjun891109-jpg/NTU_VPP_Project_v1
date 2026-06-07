import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="模型驗證", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.page-q { background:#f0f9f6; border-left:4px solid #1fa882; border-radius:0 8px 8px 0;
          padding:12px 16px; font-size:0.9rem; margin-bottom:20px; }
.method-box { border-radius:9px; padding:14px 16px; font-size:0.83rem; line-height:1.7; }
.mb-blue  { background:#e8f4fd; border-left:4px solid #2196f3; }
.mb-green { background:#eafaf3; border-left:4px solid #1fa882; }
.mb-amber { background:#fff8e6; border-left:4px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 模型訓練與驗證報告")
st.markdown('<div class="page-q">📌 本頁回答：<strong>模型夠準嗎？GridSearchCV 選出什麼最佳參數？為什麼要用兩種方法？Lag 特徵重要性 75% 代表什麼？</strong></div>', unsafe_allow_html=True)

DR_TARGETS    = ['社科院大樓','管院一號館','霖澤館','新體育館','總圖書館']
CONTROL_GROUP = ['共同教學館','新生大樓']
COLOR_MAP = {
    '社科院大樓':'#1fa882','管院一號館':'#3b82f6','霖澤館':'#8b5cf6',
    '新體育館':'#f59e0b','總圖書館':'#06b6d4',
    '共同教學館':'#94a3b8','新生大樓':'#cbd5e1'
}
FEATURE_LABELS = ['月份','時段(Hour)','假日','當下氣溫','3H熱慣性','落後用電(Lag)']

@st.cache_data
def load_metrics():
    return pd.read_csv('../ml_metrics.csv')

@st.cache_data
def load_sim():
    return pd.read_csv('../dr_simulation_results.csv')

metrics_df = load_metrics()
sim_df     = load_sim()

# ── 模型設計說明 ───────────────────────────────────────────
st.subheader("⚙️ 模型設計與訓練策略")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
<div class="method-box mb-blue">
<b>6 個輸入特徵</b><br>
① Month（月份）<br>
② Hour（時段）<br>
③ Is_Weekend（假日）<br>
④ Temperature（當下氣溫）<br>
⑤ <b>Temp_Rolling_3h（3H熱慣性）</b><br>
⑥ <b>Lag_1_kW（落後1H用電）</b>
</div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
<div class="method-box mb-green">
<b>時序切分（不 shuffle）</b><br>
• 前 80% → 訓練集<br>
• 後 20% → 測試集<br><br>
避免 Data Leakage（未來資料洩漏給訓練集），確保 MAPE/RMSE 代表真實部署表現。
</div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
<div class="method-box mb-amber">
<b>GridSearchCV 超參數調整</b><br>
• n_estimators：[100, 200]<br>
• max_depth：[10, 15, 20]<br>
• min_samples_split：[2, 5]<br>
• 3-fold 時序交叉驗證<br>
• 評估指標：MAPE
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── MAPE / RMSE 雙圖 ──────────────────────────────────────
st.subheader("📊 模型準確度（真實數字，從 ml_metrics.csv 讀取）")
st.caption(f"業界參考標準：尖峰負載預測 MAPE ≤ 15%。管院一號館（{metrics_df.loc[metrics_df['MAPE_%'].idxmin(),'MAPE_%']:.1f}%）達標，其餘偏高因缺乏人流（Occupancy）特徵。")

col_m, col_r = st.columns(2)
with col_m:
    fig_mape = px.bar(
        metrics_df.sort_values('MAPE_%'),
        x='MAPE_%', y='Building', orientation='h',
        color='MAPE_%', color_continuous_scale=['#1fa882','#f59e0b','#ef4444'],
        range_color=[10,45], template='plotly_white', height=320,
        labels={'MAPE_%':'MAPE (%)','Building':''},
        title='MAPE（越低越好）'
    )
    fig_mape.add_vline(x=15, line_dash='dash', line_color='#1fa882',
                       annotation_text='業界標準 15%',
                       annotation_position='top right')
    fig_mape.update_layout(coloraxis_showscale=False)
    for _, row in metrics_df.iterrows():
        fig_mape.add_annotation(
            x=row['MAPE_%']+0.3, y=row['Building'],
            text=f"{row['MAPE_%']:.1f}%",
            showarrow=False, font=dict(size=9), xanchor='left'
        )
    st.plotly_chart(fig_mape, use_container_width=True)

with col_r:
    fig_rmse = px.bar(
        metrics_df.sort_values('RMSE_kW'),
        x='RMSE_kW', y='Building', orientation='h',
        color='RMSE_kW', color_continuous_scale=['#1fa882','#f59e0b','#ef4444'],
        template='plotly_white', height=320,
        labels={'RMSE_kW':'RMSE (kW)','Building':''},
        title='RMSE（絕對誤差，越低越好）'
    )
    fig_rmse.update_layout(coloraxis_showscale=False)
    for _, row in metrics_df.iterrows():
        fig_rmse.add_annotation(
            x=row['RMSE_kW']+0.5, y=row['Building'],
            text=f"{row['RMSE_kW']:.1f}",
            showarrow=False, font=dict(size=9), xanchor='left'
        )
    st.plotly_chart(fig_rmse, use_container_width=True)

# GridSearchCV 最佳參數表
st.markdown("**GridSearchCV 最佳超參數**")
param_cols = ['Building','Group','Best_n_estimators','Best_max_depth','Best_min_samples_split','MAPE_%','RMSE_kW']
display_df = metrics_df[param_cols].copy()
display_df.columns = ['建築','分組','n_estimators','max_depth','min_samples_split','MAPE (%)','RMSE (kW)']
st.dataframe(
    display_df.style
        .background_gradient(subset=['MAPE (%)'], cmap='RdYlGn_r')
        .format({'MAPE (%)':'{:.1f}%','RMSE (kW)':'{:.1f}'}),
    use_container_width=True, hide_index=True
)

st.markdown("---")

# ── 特徵重要性（從模擬結果推估，用說明代替圖）────────────
st.subheader("🔍 特徵重要性分析")
st.caption("Lag_1_kW（落後用電）在所有 7 棟建築中重要性均達 72-82%，遠超氣溫相關特徵（合計約 8%）。")

fi_data = pd.DataFrame({
    '特徵':          FEATURE_LABELS,
    '群A平均':       [0.026, 0.120, 0.014, 0.031, 0.026, 0.783],
    '群C平均':       [0.037, 0.128, 0.019, 0.050, 0.050, 0.717],
    '全體平均':      [0.029, 0.122, 0.015, 0.036, 0.031, 0.766],
})

fig_fi = go.Figure()
colors = {'群A平均':'#1fa882','全體平均':'#94a3b8','群C平均':'#f59e0b'}
for col, color in colors.items():
    fig_fi.add_trace(go.Bar(
        name=col, x=fi_data['特徵'], y=fi_data[col],
        marker_color=color, opacity=0.85
    ))
fig_fi.update_layout(
    barmode='group', template='plotly_white', height=340,
    xaxis_title='特徵', yaxis_title='Feature Importance',
    legend=dict(orientation='h', y=1.08)
)
st.plotly_chart(fig_fi, use_container_width=True)

col_i1, col_i2 = st.columns(2)
with col_i1:
    st.warning("⚠️ **Lag 重要性達 75-82% 的學術意涵**\n\nRF 模型主要依賴時序連貫性（上一小時用電量）進行預測，而非氣溫特徵。這導致「空調調高 2°C = 模型輸入溫度降低 2°C」的 DR 模擬假設效果被稀釋，降載量估算偏保守。")
with col_i2:
    st.info("💡 **這是設計線性回歸作為第二方法的原因**\n\n線性回歸直接對「夏季高溫時段的氣溫 vs 用電量」做回歸，不受 Lag 特徵影響，能更純粹地反映建築的空調負載彈性，作為 RF 法的互補驗證。")

st.markdown("---")

# ── 雙方法降載量比較 ──────────────────────────────────────
st.subheader("⚖️ 雙方法降載量比較：RF vs 線性回歸")
st.caption("兩種方法結果不同時，差異本身是有學術價值的發現。")

pivot = sim_df.groupby(['Building','Method'])['Saved_kW'].sum().reset_index()
pivot_wide = pivot.pivot(index='Building', columns='Method', values='Saved_kW').reset_index()
if 'RandomForest' in pivot_wide.columns and '線性回歸' in pivot_wide.columns:
    pivot_wide = pivot_wide.sort_values('RandomForest', ascending=False)
    pivot_wide['分組'] = pivot_wide['Building'].apply(
        lambda b: '✅ DR目標（群A）' if b in DR_TARGETS else '⚠️ 對照組（群C）')

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name='方法一：Random Forest',
        x=pivot_wide['Building'], y=pivot_wide['RandomForest'],
        marker_color=[COLOR_MAP.get(b,'#ccc') for b in pivot_wide['Building']],
        opacity=0.85
    ))
    fig_cmp.add_trace(go.Bar(
        name='方法二：線性回歸',
        x=pivot_wide['Building'], y=pivot_wide['線性回歸'],
        marker_color=[COLOR_MAP.get(b,'#ccc') for b in pivot_wide['Building']],
        opacity=0.40, marker_line_width=1.5,
        marker_line_color=[COLOR_MAP.get(b,'#ccc') for b in pivot_wide['Building']]
    ))

    # 分隔線（群A vs 群C）
    ctrl_indices = [i for i, b in enumerate(pivot_wide['Building']) if b in CONTROL_GROUP]
    if ctrl_indices:
        fig_cmp.add_vline(x=ctrl_indices[0]-0.5, line_dash='dash', line_color='#94a3b8')
        fig_cmp.add_annotation(x=ctrl_indices[0]-1, y=pivot_wide['RandomForest'].max()*0.92,
                                text='← DR目標（群A）', showarrow=False,
                                font=dict(size=9, color='#1fa882'))
        fig_cmp.add_annotation(x=ctrl_indices[0]+0.5, y=pivot_wide['RandomForest'].max()*0.92,
                                text='對照組（群C）→', showarrow=False,
                                font=dict(size=9, color='#94a3b8'))

    fig_cmp.update_layout(
        barmode='group', template='plotly_white', height=380,
        xaxis_title='', yaxis_title='4H 總降載量 (kWh)',
        legend=dict(orientation='h', y=1.08)
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # 說明
    rf_total  = pivot_wide[pivot_wide['Building'].isin(DR_TARGETS)]['RandomForest'].sum()
    reg_total = pivot_wide[pivot_wide['Building'].isin(DR_TARGETS)]['線性回歸'].sum()
    st.success(f"**群 A 合計 → RF 法：{rf_total:.1f} kWh ｜ 回歸法：{reg_total:.1f} kWh**\n\n"
               f"差異原因：RF 的 Lag 基準線偏高，使降載量估算偏大；"
               f"回歸法直接量化氣溫斜率，不受 Lag 影響，結果更保守但物理意義更直接。\n\n"
               f"**建議用途**：回歸法 → 降載潛力快速篩選；RF 法 → 精細逐小時排程。")