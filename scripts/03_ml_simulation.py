import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# OS detection for correct fonts
if platform.system() == 'Windows':
    font_name = 'Microsoft JhengHei'
elif platform.system() == 'Darwin':
    font_name = 'PingFang TC'
else:
    font_name = 'sans-serif'

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=font_name)

print("Loading historical data for advanced machine learning...\n")
df = pd.read_csv('./clean_vpp_data.csv')
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values(by=['Building', 'Time'])

target_buildings = ['共同教學館', '社科院大樓', '新生大樓', '管院一號館', '霖澤館']
df_target = df[df['Building'].isin(target_buildings)].copy()

# ==========================================
# Phase 1: Feature Engineering (Time Series)
# ==========================================
print("Extracting advanced features (Thermal Inertia and Lag Variables)...")

# Feature 1: Thermal Inertia (Rolling mean of temperature over the last 3 hours)
df_target['Temp_Rolling_3h'] = df_target.groupby('Building')['Temperature'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Feature 2: Lag Load (The power consumption from the previous hour)
df_target['Lag_1_kW'] = df_target.groupby('Building')['kW'].shift(1)

# Drop rows with NaN values created by the shift operation
df_target = df_target.dropna()

# ==========================================
# Phase 2: Model Training and Validation
# ==========================================
simulation_results = []
metrics_list = []

# Scenario: Extreme heat wave in July (13:00 to 16:00)
sim_hours = [13, 14, 15, 16]
sim_temps = [36.0, 36.5, 37.0, 36.5]
sim_month = 7
sim_weekend = 0

print("Training models and validating accuracy...\n")

for b in target_buildings:
    b_data = df_target[df_target['Building'] == b].copy()
    
    # Define feature matrix and target
    features = ['Month', 'Hour', 'Is_Weekend', 'Temperature', 'Temp_Rolling_3h', 'Lag_1_kW']
    X = b_data[features]
    y = b_data['kW']
    
    # Time-series split: 80% for training, 20% for testing (no shuffle to maintain chronological order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate the model using the test set
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metrics_list.append({'Building': b, 'MAPE': f"{mape*100:.1f}%", 'RMSE': f"{rmse:.1f}"})
    
    # ==========================================
    # Phase 3: Dynamic DR Simulation
    # ==========================================
    # Get a realistic starting point for the Lag_1_kW (average load at 12:00 in summer)
    summer_noon = b_data[(b_data['Month'].isin([7, 8])) & (b_data['Hour'] == 12)]
    current_lag = summer_noon['kW'].mean() if not summer_noon.empty else y.mean()
    rolling_temp = 35.0 
    
    for i in range(4):
        # Update the simplified rolling temperature
        rolling_temp = (rolling_temp * 2 + sim_temps[i]) / 3
        
        # 1. Predict Baseline (HVAC works hard at real outdoor temperature)
        X_sim_base = pd.DataFrame([{
            'Month': sim_month, 'Hour': sim_hours[i], 'Is_Weekend': sim_weekend,
            'Temperature': sim_temps[i], 'Temp_Rolling_3h': rolling_temp, 'Lag_1_kW': current_lag
        }])
        baseline_pred = model.predict(X_sim_base)[0]
        
        # 2. Predict DR Event (Raise indoor AC by 2 degrees ≈ outdoor feels 2 degrees cooler)
        dr_temp = sim_temps[i] - 2.0
        dr_rolling = rolling_temp - 2.0
        X_sim_dr = pd.DataFrame([{
            'Month': sim_month, 'Hour': sim_hours[i], 'Is_Weekend': sim_weekend,
            'Temperature': dr_temp, 'Temp_Rolling_3h': dr_rolling, 'Lag_1_kW': current_lag
        }])
        dr_load = model.predict(X_sim_dr)[0]
        
        saved_kw = baseline_pred - dr_load
        saved_kw = max(0, saved_kw) # Prevent negative savings due to model noise
        
        simulation_results.append({
            'Building': b, 'Hour': sim_hours[i], 'Temperature': sim_temps[i],
            'Baseline_kW': baseline_pred, 'DR_Load_kW': dr_load, 'Saved_kW': saved_kw
        })
        
        # The DR load becomes the lag feature for the next hour
        current_lag = dr_load

# Display Model Accuracy
print("=== 模型準確度報告 (Model Validation) ===")
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.to_string(index=False))
print("=========================================\n")

sim_df = pd.DataFrame(simulation_results)
total_saved = sim_df['Saved_kW'].sum()
print(f"✅ 在 4 小時極端高溫中，將空調調高 2°C，目標建築共可為台大電網釋放 {total_saved:.2f} kWh 備用容量。")

# ==========================================
# Phase 4: Data Visualization
# ==========================================
plt.figure(figsize=(14, 8))
plot_df = sim_df.melt(id_vars=['Building', 'Hour'], 
                      value_vars=['Baseline_kW', 'DR_Load_kW'], 
                      var_name='Scenario', value_name='kW')

plot_df['Scenario'] = plot_df['Scenario'].replace({
    'Baseline_kW': '預期基準線 (37°C 滿載)',
    'DR_Load_kW': '執行降載 (空調調高 2°C)'
})

sns.lineplot(data=plot_df, x='Hour', y='kW', hue='Building', style='Scenario', 
             markers=True, dashes=False, linewidth=2.5, markersize=8, palette='Set2')

plt.title('【進階版】黃金降載群組：熱慣性預測與空調調控模擬', fontsize=18, pad=20)
plt.ylabel('平均用電功率 (kW)', fontsize=14)
plt.xlabel('下午時段 (13:00 - 16:00)', fontsize=14)
plt.xticks([13, 14, 15, 16])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

output_path = './dr_advanced_simulation_results.csv'
sim_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"高精度模擬數據已儲存至: {output_path}")