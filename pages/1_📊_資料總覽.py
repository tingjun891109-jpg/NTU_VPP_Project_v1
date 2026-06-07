import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="資料總覽", page_icon="📊", layout="wide")

st.markdown("""
<style>
.page-q { background:#f0f9f6; border-left:4px solid #1fa882; border-radius:0 8px 8px 0;
          padding:12px 16px; font-size:0.9rem; margin-bottom:20px; }
.story-nav { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
             padding:14px 18px; margin-bottom:20px; }
.story-prev    { color:#64748b; font-size:0.8rem; margin-bottom:4px; }
.story-finding { color:#0f7c6e; font-weight:600; font-size:0.88rem; margin:5px 0; line-height:1.5; }
.story-next    { color:#1a3a5c; font-size:0.83rem; margin-top:4px; }
.src-card { border:1px solid rgba(128,128,128,0.2); border-radius:9px;
            padding:14px 18px; font-size:0.83rem; }
.src-card h4 { margin:0 0 8px 0; font-size:0.9rem; }
.src-card li { line-height:1.8; }
</style>
""", unsafe_allow_html=True)

# ── 資料載入 ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('clean_vpp_data.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    return df

df = load_data()
all_buildings = sorted(df['Building'].unique().tolist())

# ── 頁首 + 故事線 ─────────────────────────────────────────
st.title("📊 資料總覽 & 探索性分析（EDA）")
st.markdown('<div class="page-q">本頁回答：<strong>資料長什麼樣？11 棟建築有什麼用電規律？哪些建築對氣溫最敏感？</strong></div>', unsafe_allow_html=True)

st.markdown("""
<div class="story-nav">
  <div class="story-prev">← 首頁確立了研究問題</div>
  <div class="story-finding">
    我們要精準識別「哪些建築在高溫時有空調降載彈性」，
    第一步是先理解資料的基本面貌——11 棟建築的用電規律有什麼差異？哪些對氣溫最敏感？
  </div>
  <div class="story-next">本頁的氣溫相關係數分析，將直接為下一頁的 K-Means 分群提供直覺基礎。</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── 資料來源說明 ───────────────────────────────────────────
with st.expander("📋 資料來源與 ETL 前處理說明", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="src-card">
<h4>🏗️ 台大 epower 電錶系統</h4>
<ul>
  <li>11 棟建築逐時用電功率（kW）</li>
  <li>原始格式：Excel (.xlsx)，每棟一份</li>
  <li>異常偵測：IQR 法（Q3 + 1.5×IQR）</li>
  <li>缺失值：線性插補 → ffill → bfill</li>
</ul>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="src-card">
<h4>🌡️ 中央氣象署 CODiS 逐時資料</h4>
<ul>
  <li>測站：台大校園氣象站（站號 466920）</li>
  <li>欄位：TX01 乾球溫度（°C）</li>
  <li>特殊處理：24:00 → 次日 00:00</li>
  <li>合併方式：Timestamp left join</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("**衍生特徵（Feature Engineering）**")
    feat_cols = st.columns(5)
    feats = [
        "Hour（時段）",
        "Month（月份）",
        "Is_Weekend（假日）",
        "Temp_Rolling_3h\n（3H 熱慣性均溫）",
        "Lag_1_kW\n（落後 1H 用電量）"
    ]
    for col, f in zip(feat_cols, feats):
        col.info(f)

st.markdown("---")

# ── 統計總表 ──────────────────────────────────────────────
st.subheader("🏗️ 各建築基本統計（清洗後）")

corr_map = {
    b: df[df['Building'] == b]['kW'].corr(df[df['Building'] == b]['Temperature'])
    for b in all_buildings
}
summary = df.groupby('Building').agg(
    資料筆數=('kW', 'count'),
    平均用電_kW=('kW', 'mean'),
    最大用電_kW=('kW', 'max'),
    標準差_kW=('kW', 'std'),
    平均氣溫_C=('Temperature', 'mean'),
).round(1).reset_index()
summary['氣溫相關係數_r'] = summary['Building'].map(corr_map).round(3)
summary = summary.sort_values('氣溫相關係數_r', ascending=False).reset_index(drop=True)

st.dataframe(
    summary.style
        .background_gradient(subset=['氣溫相關係數_r'], cmap='RdYlGn')
        .background_gradient(subset=['平均用電_kW'], cmap='Blues')
        .format({
            '平均用電_kW': '{:.1f}',
            '最大用電_kW': '{:.1f}',
            '標準差_kW': '{:.1f}',
            '平均氣溫_C': '{:.1f}',
            '氣溫相關係數_r': '{:.3f}'
        }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ── 用電趨勢（互動元件 1）─────────────────────────────────
st.subheader("📈 逐時用電趨勢")

col_s, col_m = st.columns([2, 2])
with col_s:
    sel = st.multiselect(
        "選擇建築（可複選）",
        all_buildings,
        default=['社科院大樓', '管院一號館', '新體育館']
        if '社科院大樓' in all_buildings else all_buildings[:3]
    )
with col_m:
    month_range = st.select_slider(
        "月份區間",
        options=list(range(1, 13)),
        value=(6, 9),
        format_func=lambda x: f"{x}月"
    )

if sel:
    dplot = df[
        (df['Building'].isin(sel)) &
        (df['Time'].dt.month >= month_range[0]) &
        (df['Time'].dt.month <= month_range[1])
    ].copy()
    dplot['日期'] = dplot['Time'].dt.date
    daily = dplot.groupby(['日期', 'Building'])['kW'].mean().reset_index()
    daily.columns = ['日期', 'Building', '日均用電_kW']

    fig_trend = px.line(
        daily, x='日期', y='日均用電_kW', color='Building',
        template='plotly_white', height=360,
        labels={'日均用電_kW': '日均用電 (kW)', '日期': ''}
    )
    fig_trend.update_layout(
        legend=dict(orientation='h', y=1.08),
        legend_title='建築'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("請選擇至少一棟建築")

st.markdown("---")

# ── 氣溫相關係數長條圖 ────────────────────────────────────
st.subheader("🌡️ 氣溫與用電量相關係數（皮爾森 r）")
st.caption("r 越高代表該建築用電量受氣溫影響越顯著，降載潛力越大。r > 0.5 為本研究的「氣候敏感」門檻。")

corr_df = summary[['Building', '氣溫相關係數_r']].copy()
corr_df['敏感等級'] = corr_df['氣溫相關係數_r'].apply(
    lambda r: '高度氣候敏感 (r > 0.5)' if r > 0.5
    else ('中度敏感 (0.35 < r ≤ 0.5)' if r > 0.35
    else '低度敏感 (r ≤ 0.35)')
)

fig_corr = px.bar(
    corr_df, x='氣溫相關係數_r', y='Building', orientation='h',
    color='敏感等級',
    color_discrete_map={
        '高度氣候敏感 (r > 0.5)': '#1fa882',
        '中度敏感 (0.35 < r ≤ 0.5)': '#f59e0b',
        '低度敏感 (r ≤ 0.35)': '#94a3b8'
    },
    template='plotly_white', height=420,
    labels={'氣溫相關係數_r': '皮爾森相關係數 r', 'Building': ''}
)
fig_corr.add_vline(
    x=0.5, line_dash='dash', line_color='#1fa882',
    annotation_text='氣候敏感門檻 r = 0.5',
    annotation_position='top right',
    annotation_font_color='#1fa882'
)
fig_corr.update_layout(
    legend=dict(orientation='h', y=1.05),
    legend_title='敏感等級'
)
st.plotly_chart(fig_corr, use_container_width=True)

st.info("💡 凝態科學館相關係數最高，但屬精密實驗設備建築（24H 運轉），不列入需量反應目標。社科院大樓、新體育館兼具高敏感度與可調空調，為核心降載候選。")

st.markdown("---")

# ── 月份 × 小時熱力圖（互動元件 2）──────────────────────
st.subheader("📅 月份 × 小時用電熱力圖")
st.caption("觀察各建築在不同月份、不同時段的用電分布，識別尖峰規律。")

default_bldg = '社科院大樓' if '社科院大樓' in all_buildings else all_buildings[0]
hm_bldg = st.selectbox("選擇建築", all_buildings,
                        index=all_buildings.index(default_bldg))

df_hm = df[df['Building'] == hm_bldg].copy()
pivot = df_hm.groupby(['Hour', 'Month'])['kW'].mean().unstack()
month_labels = ['1月', '2月', '3月', '4月', '5月', '6月',
                '7月', '8月', '9月', '10月', '11月', '12月']

fig_hm = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=month_labels,
    y=[f"{h:02d}:00" for h in pivot.index],
    colorscale='RdYlGn_r',
    colorbar=dict(title='平均 kW'),
    hovertemplate='%{y} · %{x}<br>平均：%{z:.1f} kW<extra></extra>'
))
fig_hm.update_layout(
    xaxis_title='月份',
    yaxis_title='時段',
    template='plotly_white',
    height=450,
    yaxis=dict(autorange='reversed')
)
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")
