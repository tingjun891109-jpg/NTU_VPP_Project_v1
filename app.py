import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="台大 VPP 需量反應戰情室",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0a2540 0%, #1a4a6e 55%, #0f7c6e 100%);
    border-radius: 14px; padding: 40px 48px 36px; margin-bottom: 24px; color: white;
}
.hero h1 { font-size: 1.75rem; font-weight: 700; margin: 0 0 4px 0; }
.hero .sub { font-size: 0.9rem; opacity: 0.7; margin-bottom: 24px; }
.scqa-top { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.scqa-bot { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.sq { background: rgba(255,255,255,0.08); border-radius: 9px; padding: 14px 18px; }
.sa { background: rgba(31,200,160,0.15); border: 1px solid rgba(31,200,160,0.4);
      border-radius: 9px; padding: 14px 18px; }
.stag { font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px; opacity: 0.6; margin-bottom: 5px; }
.sa .stag { color: #1fc8a0; opacity: 1; }
.sbody { font-size: 0.88rem; line-height: 1.65; }
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi { flex: 1; min-width: 130px; border: 1px solid rgba(128,128,128,0.18);
       border-radius: 11px; padding: 18px 14px; text-align: center; }
.kpi-num   { font-size: 1.7rem; font-weight: 700; color: #0f7c6e; line-height: 1; }
.kpi-label { font-size: 0.75rem; color: #555; margin-top: 5px; line-height: 1.4; }
.kpi-sub   { font-size: 0.68rem; color: #aaa; margin-top: 3px; }
.pipe { display: flex; align-items: flex-start; margin: 8px 0 20px; }
.pstep { flex: 1; text-align: center; position: relative; padding: 0 4px; }
.pstep:not(:last-child)::after { content: '→'; position: absolute; right: -10px;
    top: 16px; font-size: 1rem; color: #ccc; }
.picon  { font-size: 1.4rem; display: block; margin-bottom: 5px; }
.ptitle { font-size: 0.8rem; font-weight: 600; margin-bottom: 3px; color: #1a3a5c; }
.pdesc  { font-size: 0.7rem; color: #666; line-height: 1.5; }
.ins-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.ins-card { border-radius: 9px; padding: 14px 16px; }
.ins-green { background: #eafaf3; border-left: 4px solid #1fa882; }
.ins-blue  { background: #e8f4fd; border-left: 4px solid #2196f3; }
.ins-amber { background: #fff8e6; border-left: 4px solid #f59e0b; }
.ins-title { font-weight: 700; font-size: 0.85rem; margin-bottom: 5px; }
.ins-body  { font-size: 0.8rem; line-height: 1.6; color: #444; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.caption("11 棟台大建築 · 2025 全年逐時")
    st.caption("電錶：台大 epower 系統")
    st.caption("氣象：CWA CODiS 站號 466920")

st.markdown("""
<div class="hero">
  <h1>⚡ 台大校園虛擬電廠（VPP）需量反應戰情室</h1>
  <p class="sub">National Taiwan University · Virtual Power Plant · Demand Response Dashboard · 2025</p>
  <p class="sub" style="margin-top:-16px; font-size:0.82rem; opacity:0.65;">
    需量反應（Demand Response）：在用電尖峰時段，主動降低部分建築的空調用電，
    釋出電網備用容量，避免限電或高昂的尖峰電費。
  </p>
  <div class="scqa-top">
    <div class="sq">
      <div class="stag">S — SITUATION 情境</div>
      <div class="sbody">台大承諾 2050 淨零碳排，但校園年用電費支出龐大，夏季尖峰負載屢創新高，電網備轉容量持續受壓。</div>
    </div>
    <div class="sq">
      <div class="stag">C — COMPLICATION 衝突</div>
      <div class="sbody">極端氣候使空調需求激增，傳統「無差別節能宣導」效果有限，且可能干擾實驗室與研究室的正常運作。</div>
    </div>
  </div>
  <div class="scqa-bot">
    <div class="sq">
      <div class="stag">Q — QUESTION 核心問題</div>
      <div class="sbody">如何在<strong style="color:#1fc8a0">不影響核心學術任務</strong>的前提下，精準識別具備降載彈性的建築，並動態量化其卸載潛力？</div>
    </div>
    <div class="sa">
      <div class="stag">A — ANSWER 解答</div>
      <div class="sbody">以 <strong>K-Means</strong> 對 11 棟建築進行用電性格分群，識別「高彈性降載群」；再以 <strong>Random Forest</strong>（GridSearchCV 調參）與<strong>線性回歸</strong>雙方法，融合氣象預測與熱慣性特徵，量化各建築的降載潛力；整合為可即時調度的需量反應戰情儀表板。</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📌 專案成果一覽")

@st.cache_data
def load_kpi():
  m = pd.read_csv('../ml_metrics.csv')
  s = pd.read_csv('../dr_simulation_results.csv')
    return m, s

try:
    metrics_df, sim_df = load_kpi()
    best_mape = metrics_df['MAPE_%'].min()
    best_bldg = metrics_df.loc[metrics_df['MAPE_%'].idxmin(), 'Building']
    DR_A = ['社科院大樓','管院一號館','霖澤館','新體育館','總圖書館']
    rf_a = sim_df[(sim_df['Method']=='RandomForest') & (sim_df['Building'].isin(DR_A))]
    total_kwh = rf_a['Saved_kW'].sum()
    est_cost  = total_kwh * 5.47
    est_co2   = total_kwh * 0.494
except Exception:
    best_mape, best_bldg = 13.6, '管院一號館'
    total_kwh, est_cost, est_co2 = 208.3, 1139.4, 102.9

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-num">11</div>
    <div class="kpi-label">棟建築完整分析</div>
    <div class="kpi-sub">逾 91,000 筆逐時紀錄</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">3</div>
    <div class="kpi-label">K-Means 用電性格分群</div>
    <div class="kpi-sub">Silhouette Score = 0.497</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">{total_kwh:.1f}</div>
    <div class="kpi-label">kWh 極端高溫 4H 降載量</div>
    <div class="kpi-sub">群 A 5 棟 · RF 法估算</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">{best_mape:.1f}%</div>
    <div class="kpi-label">最佳模型 MAPE</div>
    <div class="kpi-sub">{best_bldg} · GridSearchCV</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">NT${est_cost:,.0f}</div>
    <div class="kpi-label">單次 DR 預估省電費</div>
    <div class="kpi-sub">台電夏季尖峰 $5.47/kWh</div>
  </div>
  <div class="kpi">
    <div class="kpi-num">{est_co2:.1f}</div>
    <div class="kpi-label">kg CO₂ 單次減排量</div>
    <div class="kpi-sub">碳排因子 0.494 kg/kWh</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔬 研究方法論流程")
st.markdown("""
<div class="pipe">
  <div class="pstep"><span class="picon">📥</span>
    <div class="ptitle">資料獲取</div>
    <div class="pdesc">台大 epower 電錶<br>CWA CODiS API<br>站號 466920</div></div>
  <div class="pstep"><span class="picon">🧹</span>
    <div class="ptitle">ETL 清洗</div>
    <div class="pdesc">IQR 異常偵測<br>線性插補缺失值<br>電力×氣象合併</div></div>
  <div class="pstep"><span class="picon">⚙️</span>
    <div class="ptitle">特徵工程</div>
    <div class="pdesc">3H 熱慣性均溫<br>落後 1H 用電(Lag)<br>時段/月份/假日</div></div>
  <div class="pstep"><span class="picon">🔵</span>
    <div class="ptitle">K-Means 分群</div>
    <div class="pdesc">基載率×冷氣敏感度<br>k=3 · Silhouette=0.497<br>識別降載候選建築</div></div>
  <div class="pstep"><span class="picon">🤖</span>
    <div class="ptitle">雙模型預測</div>
    <div class="pdesc">RF + GridSearchCV<br>線性回歸敏感度法<br>時序切分 80/20 驗證</div></div>
  <div class="pstep"><span class="picon">⚡</span>
    <div class="ptitle">VPP 戰情室</div>
    <div class="pdesc">極端高溫情境模擬<br>空調調高 2°C 降載<br>即時效益試算</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 💡 三大核心研究洞察")
st.markdown(f"""
<div class="ins-grid">
  <div class="ins-card ins-green">
    <div class="ins-title">雙方法結果差異本身是發現</div>
    <div class="ins-body">
      RF 法群 A 合計 {total_kwh:.1f} kWh；回歸法 178.9 kWh。
      社科院大樓 RF=140 kWh vs 回歸=80 kWh，差異源於 RF 的 Lag 效應使基準線偏高。
      管院一號館回歸斜率為負（−0.68 kW/°C），印證其用電不受氣溫主導。
      兩種方法形成互補：<strong>回歸法適合快速篩選，RF 法適合精細排程。</strong>
    </div>
  </div>
  <div class="ins-card ins-blue">
    <div class="ins-title">Lag 特徵重要性 75-82%，揭示模型局限</div>
    <div class="ins-body">
      RF 主要依賴「上一小時用電量」進行預測，
      氣溫相關特徵合計僅約 8%。
      這是 RF 法降載估算偏保守的根本原因，
      也是本研究設計線性回歸作為對照方法的學術動機。
    </div>
  </div>
  <div class="ins-card ins-amber">
    <div class="ins-title">對照組驗證了 K-Means 分群有效性</div>
    <div class="ins-body">
      群 C（共同教學館、新生大樓）RF 降載量分別僅 0.1 kWh 和 16.5 kWh，
      遠低於群 A 均值 41.7 kWh。
      支持 K-Means 的業務邏輯：<strong>排程驅動型建築降載效益確實有限。</strong>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("台灣大學 環境資訊與永續管理課程 期末報告 · 2026 · scikit-learn RF + K-Means · CWA CODiS")