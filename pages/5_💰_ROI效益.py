import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ROI 效益試算", page_icon="💰", layout="wide")

st.markdown("""
<style>
.page-q { background:#f0f9f6; border-left:4px solid #1fa882; border-radius:0 8px 8px 0;
          padding:12px 16px; font-size:0.9rem; margin-bottom:16px; }
.story-nav { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
             padding:14px 18px; margin-bottom:20px; }
.story-prev    { color:#64748b; font-size:0.8rem; margin-bottom:4px; }
.story-finding { color:#0f7c6e; font-weight:600; font-size:0.88rem;
                 margin:5px 0; line-height:1.5; }
.story-next    { color:#1a3a5c; font-size:0.83rem; margin-top:4px; }
.kpi-row { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }
.kpi { flex:1; min-width:130px; border:1px solid rgba(128,128,128,0.18);
       border-radius:11px; padding:18px 14px; text-align:center; }
.kpi-num   { font-size:1.7rem; font-weight:700; line-height:1; }
.kpi-label { font-size:0.75rem; color:#555; margin-top:5px; line-height:1.4; }
.kpi-sub   { font-size:0.68rem; color:#aaa; margin-top:3px; }
.green-n  { color:#1fa882; }
.blue-n   { color:#3b82f6; }
.amber-n  { color:#f59e0b; }
.purple-n { color:#8b5cf6; }
.result-box { border-radius:10px; padding:16px 20px; font-size:0.87rem;
              line-height:1.75; margin-top:12px; }
.rb-green { background:#eafaf3; border:1px solid rgba(31,168,130,0.3); }
.netzero-bar { background:linear-gradient(90deg,#1a4a6e,#0f7c6e);
               border-radius:8px; padding:18px 22px; color:white; margin-bottom:18px; }
.netzero-bar h3 { margin:0 0 6px 0; font-size:1rem; font-weight:700; }
.netzero-bar p  { margin:0; font-size:0.85rem; opacity:0.88; line-height:1.6; }
.cost-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:12px 0 18px; }
.cost-card { border-radius:9px; padding:14px 16px; font-size:0.83rem; line-height:1.75; }
.cost-red   { background:#fef2f2; border-left:4px solid #ef4444; }
.cost-green { background:#eafaf3; border-left:4px solid #1fa882; }
.cost-card h4 { margin:0 0 7px 0; font-size:0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── 側邊欄 ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 年度情境設定")
    st.markdown("---")
    dr_days    = st.slider("年度 DR 執行天數", 5, 60, 20,
                           help="夏季 6-9 月高溫超過 36°C 的估計天數")
    dr_hours   = st.slider("每次 DR 持續時間（小時）", 2, 6, 4)
    compliance = st.slider("系館平均配合率（%）", 30, 100, 85, 5) / 100
    method     = st.radio("降載量基準",
                          ['RF 法（208.3 kWh/次）', '回歸法（178.9 kWh/次）'])
    base_kwh   = 208.3 if 'RF' in method else 178.9
    peak_tariff = st.number_input("台電尖峰電價（NT$/kWh）", 4.0, 8.0, 5.47, 0.01)

# ── 常數 ──────────────────────────────────────────────────
EMISSION    = 0.494
TREE_ABSORB = 21.77
DR_TARGETS  = ['社科院大樓', '管院一號館', '霖澤館', '新體育館', '總圖書館']
NTU_GOAL    = 2050
NOW         = 2025

@st.cache_data
def load_sim():
    return pd.read_csv('dr_simulation_results.csv')

sim_df = load_sim()

# ── 年度效益計算 ───────────────────────────────────────────
kwh_per_event = base_kwh * (dr_hours / 4) * compliance
annual_kwh    = kwh_per_event * dr_days
annual_cost   = annual_kwh * peak_tariff
annual_co2    = annual_kwh * EMISSION
annual_trees  = annual_co2 / TREE_ABSORB
years_left    = NTU_GOAL - NOW
cum_co2       = annual_co2 * years_left

# ── 頁首 ──────────────────────────────────────────────────
st.title("ROI 效益試算 — 長期效益與成本評估")
st.markdown('<div class="page-q">本頁回答：<strong>長期來看值得做嗎？VPP 需量反應對台大淨零目標有多少貢獻？執行成本與效益如何權衡？</strong></div>', unsafe_allow_html=True)

# 故事線銜接
st.markdown(f"""
<div class="story-nav">
  <div class="story-prev">← 上一頁（VPP 戰情室）的核心發現</div>
  <div class="story-finding">
    在 37°C 極端高溫下，群 A 5 棟建築 4 小時可釋出 208.3 kWh（RF 法）或 178.9 kWh（回歸法）。
    社科院大樓單棟貢獻約 67%，管院一號館回歸斜率為負，顯示不同建築的降載特性差異顯著。
  </div>
  <div class="story-next">本頁進一步回答：這些數字換算成長期效益有多大？執行需要多少成本？是否值得投入？</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 淨零目標橫幅 ──────────────────────────────────────────
st.markdown(f"""
<div class="netzero-bar">
  <h3>台大 2050 淨零目標對應</h3>
  <p>距離 2050 淨零目標還有 {years_left} 年。
  若每年執行 {dr_days} 次需量反應，至 2050 年累積可減少碳排
  <strong>{cum_co2:,.0f} kg CO₂</strong>（約 {cum_co2/1000:.1f} 噸），
  等效種植 <strong>{cum_co2/TREE_ABSORB:,.0f} 棵樹</strong>。</p>
</div>
""", unsafe_allow_html=True)

# ── KPI ───────────────────────────────────────────────────
st.markdown("### 年度效益試算")
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-num green-n">{dr_days}</div>
    <div class="kpi-label">年度 DR 執行次數</div>
    <div class="kpi-sub">每次 {dr_hours}H · {compliance*100:.0f}% 配合率</div></div>
  <div class="kpi"><div class="kpi-num green-n">{kwh_per_event:.1f}</div>
    <div class="kpi-label">每次降載量 (kWh)</div>
    <div class="kpi-sub">基準 {base_kwh} × 時間 × 配合率</div></div>
  <div class="kpi"><div class="kpi-num green-n">{annual_kwh:,.0f}</div>
    <div class="kpi-label">年度總降載量 (kWh)</div>
    <div class="kpi-sub">= 每次 × {dr_days} 天</div></div>
  <div class="kpi"><div class="kpi-num blue-n">NT${annual_cost:,.0f}</div>
    <div class="kpi-label">年度省電費</div>
    <div class="kpi-sub">{peak_tariff:.2f} NT$/kWh</div></div>
  <div class="kpi"><div class="kpi-num amber-n">{annual_co2:,.0f}</div>
    <div class="kpi-label">年度減碳量 (kg CO₂)</div>
    <div class="kpi-sub">碳排因子 {EMISSION} kg/kWh</div></div>
  <div class="kpi"><div class="kpi-num purple-n">{annual_trees:,.0f}</div>
    <div class="kpi-label">等效種樹數（棵/年）</div>
    <div class="kpi-sub">一棵樹年吸碳 {TREE_ABSORB} kg</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 累積效益曲線 ──────────────────────────────────────────
st.subheader("累積效益預測（2025-2050）")

years   = list(range(NOW, NTU_GOAL + 1))
cum_kwh = [annual_kwh * (y-NOW+1) for y in years]
cum_co2_arr = [annual_co2 * (y-NOW+1) for y in years]

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=years, y=[c/1000 for c in cum_kwh], name='累積降載量 (MWh)',
    mode='lines', line=dict(color='#1fa882', width=2.5),
    fill='tozeroy', fillcolor='rgba(31,168,130,0.1)',
    hovertemplate='%{x} 年<br>累積降載：%{y:.1f} MWh<extra></extra>'
))
fig_cum.add_trace(go.Scatter(
    x=years, y=[c/1000 for c in cum_co2_arr], name='累積減碳量 (ton CO₂)',
    mode='lines', line=dict(color='#3b82f6', width=2.5, dash='dash'),
    yaxis='y2',
    hovertemplate='%{x} 年<br>累積減碳：%{y:.2f} 噸 CO₂<extra></extra>'
))
for yr, label in {2030:'2030 中期目標', 2040:'2040 電氣化', 2050:'2050 淨零'}.items():
    if yr in years:
        idx = years.index(yr)
        fig_cum.add_vline(x=yr, line_dash='dot', line_color='#94a3b8', line_width=1.2)
        fig_cum.add_annotation(x=yr, y=cum_kwh[idx]/1000*0.8, text=label,
                                showarrow=False, font=dict(size=8.5, color='#64748b'),
                                bgcolor='rgba(255,255,255,0.85)', borderpad=3)
fig_cum.update_layout(
    xaxis_title='年份', template='plotly_white', height=350,
    yaxis =dict(title='累積降載量 (MWh)', color='#1fa882'),
    yaxis2=dict(title='累積減碳量 (ton CO₂)', color='#3b82f6',
                overlaying='y', side='right'),
    legend=dict(orientation='h', y=1.08), hovermode='x unified'
)
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---")

# ── 成本 vs 效益 ──────────────────────────────────────────
st.subheader("成本 vs 效益：完整 ROI 評估")
st.caption("僅量化效益而忽略成本，不構成完整的 ROI 分析。以下列出執行 VPP 的主要成本項目。")

st.markdown(f"""
<div class="cost-grid">
  <div class="cost-card cost-red">
    <h4>執行成本（估算）</h4>
    <b>一次性建置成本：</b><br>
    · 智慧電錶升級與通訊模組：NT$50-100 萬<br>
    · VPP 調度系統軟體開發：NT$30-80 萬<br>
    · 初期人員培訓：NT$5-10 萬<br><br>
    <b>年度運營成本：</b><br>
    · 系統維護與管理人力：NT$10-20 萬/年<br>
    · 各系館協調溝通成本：難以量化<br>
    · 研究室短暫不便（機會成本）：難以量化
  </div>
  <div class="cost-card cost-green">
    <h4>效益（本研究量化）</h4>
    <b>直接效益：</b><br>
    · 年度省電費：NT${annual_cost:,.0f}<br>
    · 年度減碳：{annual_co2:,.0f} kg CO₂<br><br>
    <b>間接效益（未量化）：</b><br>
    · 碳權價值（若未來納入碳交易市場）<br>
    · 台電需量反應獎勵費率<br>
    · 校園能源管理能力建構<br>
    · 淨零形象與人才招募優勢
  </div>
</div>
""", unsafe_allow_html=True)

# 回收期計算
st.markdown("**簡易回收期試算**")
col_r1, col_r2 = st.columns(2)
with col_r1:
    setup_cost = st.number_input("一次性建置成本（萬 NT$）", 50, 300, 150, 10)
with col_r2:
    annual_ops = st.number_input("年度運營成本（萬 NT$）", 5, 50, 15, 5)

setup_nt   = setup_cost * 10000
ops_nt     = annual_ops * 10000
net_annual = annual_cost - ops_nt

if net_annual > 0:
    payback = setup_nt / net_annual
    st.success(f"**回收期估算：約 {payback:.1f} 年**（建置成本 NT${setup_cost}萬 ÷ 年度淨效益 NT${net_annual:,.0f}）\n\n"
               f"若計入台電需量反應獎勵與碳權價值，回收期可能進一步縮短。")
else:
    st.error(f"目前參數下年度運營成本（NT${ops_nt:,.0f}）高於省電效益（NT${annual_cost:,.0f}），"
             f"請提高 DR 天數或配合率。")

# 回收期 vs DR 天數折線
days_arr   = list(range(5, 61, 5))
pb_arr     = []
for d in days_arr:
    net = base_kwh*(dr_hours/4)*compliance*d*peak_tariff - ops_nt
    pb_arr.append(round(setup_nt/net, 1) if net > 0 else None)

pb_df = pd.DataFrame({'DR執行天數': days_arr, '回收期(年)': pb_arr}).dropna()
if not pb_df.empty:
    fig_pb = px.line(pb_df, x='DR執行天數', y='回收期(年)', markers=True,
                     template='plotly_white', height=270,
                     title=f'建置成本 NT${setup_cost}萬 · 年度運營 NT${annual_ops}萬',
                     labels={'DR執行天數':'年度 DR 執行天數', '回收期(年)':'回收期（年）'})
    fig_pb.add_hline(y=10, line_dash='dash', line_color='#f59e0b',
                     annotation_text='10 年回收門檻')
    fig_pb.add_hline(y=5, line_dash='dash', line_color='#1fa882',
                     annotation_text='5 年回收門檻')
    fig_pb.update_traces(line_color='#1a4a6e', marker_color='#1fa882')
    st.plotly_chart(fig_pb, use_container_width=True)

st.markdown("---")

# ── 敏感度分析（修正版：三圖各自獨立）────────────────────
st.subheader("敏感度分析：各參數對年度省電費的影響")
st.caption("每張圖只改變一個參數，其餘固定在預設值，確保比較具有意義。紅線 = 目前設定值。")

BASE = dict(days=20, hours=4, comp=0.85, kwh=208.3, tariff=5.47)

tab1, tab2, tab3 = st.tabs(['DR 執行天數（固定其他）',
                              '系館配合率（固定其他）',
                              '每次持續時間（固定其他）'])
with tab1:
    x1 = list(range(5, 61, 5))
    y1 = [BASE['kwh']*(BASE['hours']/4)*BASE['comp']*d*BASE['tariff'] for d in x1]
    fig1 = go.Figure(go.Bar(x=x1, y=y1, marker_color='#1fa882', opacity=0.82))
    fig1.add_vline(x=dr_days, line_dash='dash', line_color='#e11d48',
                   annotation_text=f'目前設定：{dr_days} 天',
                   annotation_position='top right',
                   annotation_font_color='#e11d48')
    fig1.update_layout(xaxis_title='年度 DR 執行天數',
                       yaxis_title='年度省電費（NT$）',
                       template='plotly_white', height=290,
                       xaxis=dict(tickmode='linear', dtick=5))
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(f"固定條件：每次 {BASE['hours']}H · 配合率 {BASE['comp']*100:.0f}% · 電價 NT${BASE['tariff']}/kWh")

with tab2:
    x2 = list(range(30, 101, 5))
    y2 = [BASE['kwh']*(BASE['hours']/4)*(c/100)*BASE['days']*BASE['tariff'] for c in x2]
    fig2 = go.Figure(go.Bar(x=x2, y=y2, marker_color='#3b82f6', opacity=0.82))
    fig2.add_vline(x=compliance*100, line_dash='dash', line_color='#e11d48',
                   annotation_text=f'目前設定：{compliance*100:.0f}%',
                   annotation_position='top right',
                   annotation_font_color='#e11d48')
    fig2.update_layout(xaxis_title='系館配合率（%）',
                       yaxis_title='年度省電費（NT$）',
                       template='plotly_white', height=290,
                       xaxis=dict(tickmode='linear', dtick=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"固定條件：DR {BASE['days']} 天/年 · 每次 {BASE['hours']}H · 電價 NT${BASE['tariff']}/kWh")

with tab3:
    x3 = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    y3 = [BASE['kwh']*(h/4)*BASE['comp']*BASE['days']*BASE['tariff'] for h in x3]
    fig3 = go.Figure(go.Bar(x=x3, y=y3, marker_color='#f59e0b', opacity=0.82))
    fig3.add_vline(x=dr_hours, line_dash='dash', line_color='#e11d48',
                   annotation_text=f'目前設定：{dr_hours}H',
                   annotation_position='top right',
                   annotation_font_color='#e11d48')
    fig3.update_layout(xaxis_title='每次 DR 持續時間（小時）',
                       yaxis_title='年度省電費（NT$）',
                       template='plotly_white', height=290,
                       xaxis=dict(tickmode='linear', dtick=0.5))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"固定條件：DR {BASE['days']} 天/年 · 配合率 {BASE['comp']*100:.0f}% · 電價 NT${BASE['tariff']}/kWh")

st.markdown("---")

# ── 各建築貢獻 ─────────────────────────────────────────────
st.subheader("各建築年度效益貢獻")

bldg_saved = (sim_df[(sim_df['Method']=='RandomForest') &
                      (sim_df['Building'].isin(DR_TARGETS))]
              .groupby('Building')['Saved_kW'].sum())
bldg_annual = pd.DataFrame({
    '建築':            bldg_saved.index,
    '每次降載 (kWh)':  (bldg_saved*(dr_hours/4)*compliance).round(1).values,
    '年度省電費 (NT$)': (bldg_saved*(dr_hours/4)*compliance*dr_days*peak_tariff).round(0).astype(int).values,
    '年度減碳 (kg)':   (bldg_saved*(dr_hours/4)*compliance*dr_days*EMISSION).round(1).values,
    '貢獻佔比 (%)':    (bldg_saved/bldg_saved.sum()*100).round(1).values,
}).sort_values('每次降載 (kWh)', ascending=False)

col_t, col_p = st.columns([1.3, 1])
with col_t:
    st.dataframe(
        bldg_annual.style
            .background_gradient(subset=['每次降載 (kWh)'], cmap='Greens')
            .format({'每次降載 (kWh)':'{:.1f}',
                     '年度省電費 (NT$)':'NT$ {:,}',
                     '年度減碳 (kg)':'{:.1f}',
                     '貢獻佔比 (%)':'{:.1f}%'}),
        use_container_width=True, hide_index=True
    )
with col_p:
    COLOR_MAP = {'社科院大樓':'#1fa882','管院一號館':'#3b82f6',
                 '霖澤館':'#8b5cf6','新體育館':'#f59e0b','總圖書館':'#06b6d4'}
    fig_pie = px.pie(bldg_annual, values='每次降載 (kWh)', names='建築',
                     hole=0.45, template='plotly_white',
                     color_discrete_map=COLOR_MAP)
    fig_pie.update_traces(textinfo='percent+label', textfont_size=11)
    fig_pie.update_layout(height=290, showlegend=False,
                          margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ── 研究限制與未來方向 ────────────────────────────────────
st.subheader("研究限制與未來研究方向")
col_l, col_f = st.columns(2)
with col_l:
    st.error("""**研究限制**

1. 缺乏 Occupancy 特徵：門禁人流資料可顯著提升共同教學館與新生大樓的預測準確度

2. Lag 特徵主導 RF 模型（75-82%）：氣溫介入效果被稀釋，降載量估算偏保守

3. 模擬假設簡化：空調調高 2°C ≈ 外溫降低 2°C，忽略建築熱響應時延

4. K-Means 樣本數偏少（11 棟），Silhouette Score 0.497 屬中等品質
""")
with col_f:
    st.info("""**未來研究方向**

1. 整合台大門禁系統（Occupancy）作為核心特徵，提升排程驅動型建築的預測力

2. 結合台電時間電價（TOU）動態排程，在電費最高時段優先執行 DR

3. 以強化學習（RL）代替固定規則，動態決定最佳降載組合

4. 擴大至全校 100+ 棟建築，評估校園整體 VPP 潛力上限
""")

# ── 結論 ──────────────────────────────────────────────────
net_str = f"NT${(annual_cost - 15*10000):,.0f}" if (annual_cost - 15*10000) > 0 else "（詳見上方試算）"
st.markdown(f"""
<div class="result-box rb-green">
<strong>研究結論</strong><br>
台大校園 VPP 需量反應在技術上可行，且具備明確的長期效益。
以群 A 5 棟建築為核心，每年執行 {dr_days} 次、每次 {dr_hours} 小時的需量反應，
預估年度省電費 <strong>NT${annual_cost:,.0f}</strong>，年度減碳 <strong>{annual_co2:,.0f} kg CO₂</strong>。
至 2050 年累積減碳可達 <strong>{cum_co2/1000:.1f} 噸 CO₂</strong>。<br><br>
成本面分析顯示，在合理建置成本下，回收期約 5-15 年（依執行頻率與電價而定），
若計入台電需量反應獎勵與未來碳權價值，財務可行性將顯著提升。<br><br>
本研究同時揭示：Occupancy 特徵是提升模型準確度的最關鍵突破口；
VPP 框架提供一個可擴展的校園能源管理平台，是台大邁向 2050 淨零的具體可行路徑。
</div>
""", unsafe_allow_html=True)

st.markdown("---")
