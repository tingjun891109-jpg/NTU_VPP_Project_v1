import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="VPP 戰情室", page_icon="🎯", layout="wide")
st.title("🎯 VPP 需量反應戰情室 — 互動降載模擬")
st.caption("調整右側參數，系統即時計算降載容量、節省電費與碳排減量。")

@st.cache_data
def load_sim():
    return pd.read_csv('./dr_advanced_simulation_results.csv')

df_sim = load_sim()

# ── 台大建築 GPS 座標 ───────────────────────────────────────
BUILDING_COORDS = {
    '共同教學館': [25.0165, 121.5375],
    '社科院大樓': [25.0205, 121.5420],
    '新生大樓':   [25.0180, 121.5360],
    '管院一號館': [25.0135, 121.5365],
    '霖澤館':     [25.0210, 121.5435],
}
# 台電尖峰電價（元/kWh，夏季 6-9 月）
PEAK_TARIFF = 5.47
# 碳排因子（kg CO2/kWh，2024 台電年報）
EMISSION_FACTOR = 0.494

# ── 側邊欄互動元件 ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ 模擬參數設定")
    st.markdown("---")

    # 互動元件 1：建築選擇
    selected_buildings = st.multiselect(
        "🏗️ 選擇降載目標建築",
        options=list(BUILDING_COORDS.keys()),
        default=list(BUILDING_COORDS.keys()),
        help="僅「氣候敏感型」建築可進行需量反應"
    )

    # 互動元件 2：氣溫滑桿
    sim_temp = st.slider(
        "🌡️ 模擬極端氣溫 (°C)",
        min_value=28, max_value=38, value=37, step=1,
        help="滑桿右移代表更極端的高溫情境"
    )

    # 互動元件 3：降載小時數
    dr_hours = st.slider(
        "⏱️ 需量反應持續時間（小時）",
        min_value=1, max_value=6, value=4
    )

    # 互動元件 4：空調調升溫度
    ac_delta = st.select_slider(
        "❄️ 空調設定溫度調高",
        options=[1, 1.5, 2, 2.5, 3],
        value=2,
        format_func=lambda x: f"+{x}°C"
    )

    st.markdown("---")
    st.caption(f"電價：NT$ {PEAK_TARIFF}/kWh（台電夏季尖峰）")
    st.caption(f"碳排因子：{EMISSION_FACTOR} kg CO₂/kWh")

# ── 主要佈局 ───────────────────────────────────────────────
col_map, col_chart = st.columns([1.1, 1])

with col_map:
    st.subheader("📍 台大校園降載資源地圖")

    # 動態調整降載量（依氣溫與小時數線性縮放）
    temp_factor = (sim_temp - 28) / (38 - 28)
    hour_factor = dr_hours / 4

    m = folium.Map(location=[25.0173, 121.5397], zoom_start=16, tiles='OpenStreetMap')

    for bldg in selected_buildings:
        if bldg not in BUILDING_COORDS:
            continue
        coords = BUILDING_COORDS[bldg]
        b_data = df_sim[df_sim['Building'] == bldg]
        base_saved = b_data['Saved_kW'].sum() if not b_data.empty else 20
        adj_saved = base_saved * temp_factor * hour_factor * (ac_delta / 2)

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px">
          <b>{bldg}</b><br>
          <span style="color:#1fa882">預估降載：{adj_saved:.1f} kWh</span><br>
          預估省電費：NT$ {adj_saved * PEAK_TARIFF:.0f}<br>
          減碳：{adj_saved * EMISSION_FACTOR:.1f} kg CO₂
        </div>"""
        folium.CircleMarker(
            location=coords,
            radius=8 + adj_saved * 0.08,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{bldg}｜{adj_saved:.1f} kWh",
            color='#1fa882', fill=True, fill_color='#1fa882', fill_opacity=0.75
        ).add_to(m)

    st_folium(m, width=520, height=430, returned_objects=[])

with col_chart:
    st.subheader("📈 逐時降載效益分析")

    if not selected_buildings:
        st.warning("請在左側選擇至少一棟建築")
    else:
        bldg_chart = st.selectbox("分析特定建築", selected_buildings)
        plot_data = df_sim[df_sim['Building'] == bldg_chart].copy()

        if not plot_data.empty:
            # 依氣溫動態調整降載量
            plot_data = plot_data.copy()
            scale = temp_factor * hour_factor * (ac_delta / 2)
            plot_data['DR_Load_adj'] = plot_data['Baseline_kW'] - (
                (plot_data['Baseline_kW'] - plot_data['DR_Load_kW']) * scale
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data['Hour'], y=plot_data['Baseline_kW'],
                name=f'基準線（{sim_temp}°C）',
                line=dict(color='#e74c3c', width=2.5),
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=plot_data['Hour'], y=plot_data['DR_Load_adj'],
                name=f'執行降載（調高 {ac_delta}°C）',
                line=dict(color='#1fa882', width=2.5, dash='dash'),
                mode='lines+markers'
            ))
            fig.add_vrect(x0=10, x1=10+dr_hours,
                          fillcolor='rgba(31,168,130,0.1)',
                          annotation_text="DR 執行時段",
                          annotation_position="top left",
                          line_width=0)
            fig.update_layout(
                xaxis_title='時間（小時）',
                yaxis_title='用電功率 (kW)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                template='plotly_white', height=380
            )
            st.plotly_chart(fig, use_container_width=True)

# ── 效益試算總結 ───────────────────────────────────────────
st.markdown("---")
st.subheader("💰 需量反應效益試算")

total_saved_kwh = 0
rows = []
for bldg in selected_buildings:
    if bldg not in BUILDING_COORDS:
        continue
    b_data = df_sim[df_sim['Building'] == bldg]
    base_saved = b_data['Saved_kW'].sum() if not b_data.empty else 20
    adj = base_saved * temp_factor * hour_factor * (ac_delta / 2)
    cost = adj * PEAK_TARIFF
    carbon = adj * EMISSION_FACTOR
    total_saved_kwh += adj
    rows.append({'建築': bldg, '降載容量 (kWh)': f"{adj:.1f}",
                 '省電費 (NT$)': f"{cost:.0f}", '減碳 (kg CO₂)': f"{carbon:.1f}"})

if rows:
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🔋 總降載容量", f"{total_saved_kwh:.1f} kWh")
    with m2:
        st.metric("💵 預估省電費", f"NT$ {total_saved_kwh * PEAK_TARIFF:,.0f}")
    with m3:
        st.metric("🌱 減少碳排", f"{total_saved_kwh * EMISSION_FACTOR:.1f} kg CO₂")
    with m4:
        eq_trees = total_saved_kwh * EMISSION_FACTOR / 21.77  # 一棵樹年吸碳量
        st.metric("🌳 等效種樹", f"{eq_trees:.1f} 棵")

    st.success(f"✅ 在 **{sim_temp}°C** 極端高溫下，{len(selected_buildings)} 棟建築執行 **{dr_hours} 小時**需量反應，空調調高 **{ac_delta}°C**，預估可為台大電網釋出 **{total_saved_kwh:.1f} kWh** 備用容量。")