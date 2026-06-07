import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="VPP 戰情室", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.page-q { background:#f0f9f6; border-left:4px solid #1fa882; border-radius:0 8px 8px 0;
          padding:12px 16px; font-size:0.9rem; margin-bottom:20px; }
.kpi-row { display:flex; gap:10px; margin:12px 0; flex-wrap:wrap; }
.kpi { flex:1; min-width:110px; border-radius:10px; padding:14px 12px; text-align:center;
       border:1px solid rgba(128,128,128,0.15); }
.kpi-num   { font-size:1.5rem; font-weight:700; line-height:1; }
.kpi-label { font-size:0.73rem; color:gray; margin-top:4px; }
.green-n  { color:#1fa882; }
.blue-n   { color:#3b82f6; }
.amber-n  { color:#f59e0b; }
.purple-n { color:#8b5cf6; }
.gray-n   { color:#6b7280; }
.result-box { border-radius:10px; padding:14px 18px; font-size:0.85rem; line-height:1.7; margin-top:12px; }
.rb-green { background:#eafaf3; border:1px solid rgba(31,168,130,0.3); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 情境模擬參數")
    st.markdown("---")

    # 互動元件 1：方法選擇
    method = st.radio("**預測方法**",
                      options=['RandomForest（預設）','線性回歸（對照）'],
                      index=0,
                      help="RF 法偏保守但精細；回歸法更直接反映空調敏感度")
    method_key = 'RandomForest' if 'Random' in method else '線性回歸'

    st.markdown("---")

    # 互動元件 2：建築多選
    DR_TARGETS    = ['社科院大樓','管院一號館','霖澤館','新體育館','總圖書館']
    CONTROL_GROUP = ['共同教學館','新生大樓']
    ALL_BUILDINGS = DR_TARGETS + CONTROL_GROUP

    st.markdown("**建築選擇**")
    show_ctrl = st.checkbox("顯示對照組（群C）", value=False)
    options   = ALL_BUILDINGS if show_ctrl else DR_TARGETS
    selected  = st.multiselect("選擇建築", options=options,
                               default=DR_TARGETS,
                               help="★ = 群A正式DR目標")

    st.markdown("---")

    # 互動元件 3：情境滑桿
    st.markdown("**情境調整**")
    sim_temp   = st.slider("模擬氣溫 (°C)", 28, 38, 37, 1)
    ac_delta   = st.select_slider("空調調高幅度",
                                  options=[1.0, 1.5, 2.0, 2.5, 3.0], value=2.0,
                                  format_func=lambda x: f"+{x:.1f}°C")
    compliance = st.slider("系館配合率 (%)", 30, 100, 90, 5) / 100

    if sim_temp >= 36:
        st.error(f"🌡️ {sim_temp}°C — 極端高溫警報")
    elif sim_temp >= 33:
        st.warning(f"🌡️ {sim_temp}°C — 需量反應建議")
    else:
        st.info(f"🌡️ {sim_temp}°C — 一般情境")

    st.caption(f"電價：NT$5.47/kWh · 碳排：0.494 kg/kWh")

# ── 常數 ──────────────────────────────────────────────────
PEAK_TARIFF     = 5.47
EMISSION_FACTOR = 0.494
TREE_ABSORB     = 21.77

BUILDING_COORDS = {
    '社科院大樓': [25.0205, 121.5420],
    '管院一號館': [25.0135, 121.5365],
    '霖澤館':    [25.0210, 121.5435],
    '新體育館':  [25.0178, 121.5410],
    '總圖書館':  [25.0170, 121.5390],
    '共同教學館': [25.0165, 121.5375],
    '新生大樓':  [25.0180, 121.5360],
}
COLOR_MAP = {
    '社科院大樓':'#1fa882','管院一號館':'#3b82f6','霖澤館':'#8b5cf6',
    '新體育館':'#f59e0b','總圖書館':'#06b6d4',
    '共同教學館':'#94a3b8','新生大樓':'#cbd5e1',
}

# ── 頁首 ──────────────────────────────────────────────────
st.title("VPP 需量反應互動戰情室")
st.markdown('<div class="page-q">本頁回答：<strong>如果今天執行需量反應，能省多少電？調整不同情境參數（氣溫、空調幅度、配合率），即時查看降載效益。</strong></div>', unsafe_allow_html=True)

st.markdown("""
<div class="story-nav">
  <div class="story-prev">← 上一頁（模型驗證）的核心發現</div>
  <div class="story-finding">
    管院一號館 MAPE=13.6% 最優；RF 模型的 Lag 特徵重要性達 75-82%，
    導致降載估算偏保守。線性回歸法則更直接反映氣溫敏感度，兩法互補。
  </div>
  <div class="story-next">本頁進一步回答：在實際情境下，輸入不同氣溫與空調條件，各建築能釋出多少電網容量？</div>
</div>
""", unsafe_allow_html=True)

# ── 載入模擬資料 ──────────────────────────────────────────
@st.cache_data
def load_sim():
    return pd.read_csv('../dr_simulation_results.csv')

sim_df = load_sim()

# ── 動態計算（依滑桿縮放）────────────────────────────────
temp_factor  = max(0, (sim_temp - 28) / (37 - 28))
delta_factor = ac_delta / 2.0

rows = []
for b in selected:
    bdata = sim_df[(sim_df['Building']==b) & (sim_df['Method']==method_key)]
    for _, r in bdata.iterrows():
        base    = r['Baseline_kW'] * temp_factor
        saved   = r['Saved_kW'] * temp_factor * delta_factor * compliance
        dr_load = max(0, base - saved)
        rows.append({'Building':b, 'Hour':r['Hour'], 'Temperature':r['Temperature'],
                     'Baseline_kW':base, 'DR_Load_kW':dr_load, 'Saved_kW':saved})

calc_df = pd.DataFrame(rows) if rows else pd.DataFrame(
    columns=['Building','Hour','Baseline_kW','DR_Load_kW','Saved_kW'])

total_saved  = calc_df['Saved_kW'].sum()
total_cost   = total_saved * PEAK_TARIFF
total_carbon = total_saved * EMISSION_FACTOR
total_trees  = total_carbon / TREE_ABSORB

# ── KPI 橫幅 ──────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-num green-n">{total_saved:.1f}</div>
    <div class="kpi-label">總降載量 (kWh)</div></div>
  <div class="kpi"><div class="kpi-num blue-n">NT${total_cost:,.0f}</div>
    <div class="kpi-label">預估省電費</div></div>
  <div class="kpi"><div class="kpi-num amber-n">{total_carbon:.1f} kg</div>
    <div class="kpi-label">CO₂ 減排量</div></div>
  <div class="kpi"><div class="kpi-num purple-n">{total_trees:.1f} 棵</div>
    <div class="kpi-label">等效種樹數</div></div>
  <div class="kpi"><div class="kpi-num gray-n">{len(selected)}</div>
    <div class="kpi-label">參與建築棟數</div></div>
  <div class="kpi"><div class="kpi-num gray-n">{compliance*100:.0f}%</div>
    <div class="kpi-label">系館配合率</div></div>
</div>
""", unsafe_allow_html=True)

st.caption(
    "注意：氣溫與空調幅度的縮放採線性近似（相對於原始模擬情境 37°C、+2°C），"
    "僅供情境探索，精確逐小時預測值請參考「模型驗證」頁面。"
)
st.markdown("---")

# ── 地圖 + 折線圖 ─────────────────────────────────────────
map_col, chart_col = st.columns([1.1, 1])

with map_col:
    st.subheader("📍 台大校園降載資源地圖")
    per_bldg = calc_df.groupby('Building')['Saved_kW'].sum().to_dict()

    m = folium.Map(location=[25.0178, 121.5395], zoom_start=16,
                   tiles='CartoDB positron')

    for bldg, coords in BUILDING_COORDS.items():
        saved     = per_bldg.get(bldg, 0)
        is_sel    = bldg in selected
        is_target = bldg in DR_TARGETS
        color     = COLOR_MAP.get(bldg, '#ccc')
        radius    = max(7, min(28, 7 + saved * 0.12)) if is_sel else 6
        opacity   = 0.85 if is_sel else 0.3

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:160px;font-size:13px">
          <b>{'★ ' if is_target else ''}{bldg}</b><br>
          {'<span style="color:#1fa882"><b>預估降載：' + f'{saved:.1f} kWh</b></span><br>' if is_sel else '<span style="color:#999">未納入本次模擬</span><br>'}
          {'省電費：NT$ ' + f'{saved*PEAK_TARIFF:,.0f}' if is_sel and saved>0 else ''}
        </div>"""

        folium.CircleMarker(
            location=coords, radius=radius,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{'★ ' if is_target else ''}{'✅' if is_sel else '○'} {bldg}",
            color='white', weight=1.5,
            fill=True, fill_color=color, fill_opacity=opacity
        ).add_to(m)

    st_folium(m, width=None, height=420, returned_objects=[])

with chart_col:
    st.subheader("📈 逐時降載曲線")
    if calc_df.empty or not selected:
        st.info("請在左側選擇建築。")
    else:
        chart_bldg = st.selectbox("分析建築", options=selected)
        bplot = calc_df[calc_df['Building']==chart_bldg].sort_values('Hour')

        fig_line = go.Figure()
        color = COLOR_MAP.get(chart_bldg, '#1fa882')

        fig_line.add_trace(go.Scatter(
            x=bplot['Hour'], y=bplot['Baseline_kW'],
            name=f'基準線（{sim_temp}°C）',
            mode='lines+markers',
            line=dict(color='#e11d48', width=2.5),
            marker=dict(size=9),
            hovertemplate='%{x}:00｜基準 %{y:.1f} kW<extra></extra>'
        ))
        fig_line.add_trace(go.Scatter(
            x=bplot['Hour'], y=bplot['DR_Load_kW'],
            name=f'執行降載（+{ac_delta}°C，配合率{compliance*100:.0f}%）',
            mode='lines+markers',
            line=dict(color=color, width=2.5, dash='dash'),
            marker=dict(size=9, symbol='diamond'),
            hovertemplate='%{x}:00｜降載後 %{y:.1f} kW<extra></extra>'
        ))
        # 填充降載區域
        fig_line.add_traces([go.Scatter(
            x=list(bplot['Hour']) + list(bplot['Hour'])[::-1],
            y=list(bplot['Baseline_kW']) + list(bplot['DR_Load_kW'])[::-1],
            fill='toself', fillcolor=f'rgba(31,168,130,0.10)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'
        )])
        fig_line.update_layout(
            xaxis=dict(tickmode='linear', dtick=1, title='時段（小時）'),
            yaxis_title='用電功率 (kW)',
            template='plotly_white', height=340,
            legend=dict(orientation='h', y=1.08, font=dict(size=10)),
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)

        if chart_bldg == '共同教學館':
            st.warning("⚠️ 共同教學館降載效益微小（原始模擬值 ≈ 0.1 kWh），用電受課表排程驅動而非氣溫。詳見模型驗證頁。")
        if chart_bldg == '管院一號館' and method_key == '線性回歸':
            st.warning("⚠️ 管院一號館回歸斜率為負（−0.68 kW/°C），回歸法降載量為 0。RF 法仍有少量降載，兩法差異為本研究發現之一。")

st.markdown("---")

# ── 效益明細表 ─────────────────────────────────────────────
st.subheader("💰 各建築效益明細")

if not calc_df.empty:
    summary_rows = []
    for b in selected:
        bdata = calc_df[calc_df['Building']==b]
        saved = bdata['Saved_kW'].sum()
        summary_rows.append({
            '建築': b,
            '分組': '✅ DR目標（群A）' if b in DR_TARGETS else '⚠️ 對照組（群C）',
            '4H 降載量 (kWh)': round(saved, 2),
            '省電費 (NT$)': round(saved * PEAK_TARIFF),
            '減碳 (kg CO₂)': round(saved * EMISSION_FACTOR, 1),
            '貢獻佔比 (%)': round(saved/total_saved*100, 1) if total_saved > 0 else 0
        })
    summary_tbl = pd.DataFrame(summary_rows).sort_values('4H 降載量 (kWh)', ascending=False)

    st.dataframe(
        summary_tbl.style
            .format({'4H 降載量 (kWh)':'{:.2f}','省電費 (NT$)':'NT$ {:,}',
                     '減碳 (kg CO₂)':'{:.1f}','貢獻佔比 (%)':'{:.1f}%'})
            .apply(lambda row: ['background-color: rgba(31,168,130,0.06)']*len(row)
                   if '✅' in str(row['分組']) else ['']*len(row), axis=1),
        use_container_width=True, hide_index=True
    )

    best = summary_tbl.iloc[0]
    method_label = 'RF 法' if method_key == 'RandomForest' else '回歸法'
    st.markdown(f"""
<div class="result-box rb-green">
⚡ <b>本次模擬結論</b>（{sim_temp}°C · 空調調高 {ac_delta}°C · 配合率 {compliance*100:.0f}% · {method_label}）<br>
{len(selected)} 棟建築執行 4 小時需量反應，合計可為台大電網釋出 <b>{total_saved:.1f} kWh</b> 備用容量，
預估省電費 <b>NT$ {total_cost:,.0f}</b>，減少碳排 <b>{total_carbon:.1f} kg CO₂</b>（等效種植 {total_trees:.1f} 棵樹）。<br>
最大貢獻者：<b>{best['建築']}</b>（{best['4H 降載量 (kWh)']:.1f} kWh，佔 {best['貢獻佔比 (%)']:.0f}%）。
</div>
""", unsafe_allow_html=True)