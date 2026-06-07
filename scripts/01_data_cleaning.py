import pandas as pd
import glob
import os
import platform

# ==========================================
# 0. 環境與路徑設定 (本機版)
# ==========================================
# 自動偵測系統並設定中文字型
if platform.system() == 'Windows':
    font_name = 'Microsoft JhengHei' # 微軟正黑體
elif platform.system() == 'Darwin':
    font_name = 'PingFang TC'        # 蘋果蘋方體
else:
    font_name = 'sans-serif'

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=font_name)

# 設定相對路徑 (假設程式碼與 raw_data 資料夾在同一層)
folder_path = './raw_data/'
excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

print("開始執行本機端資料 ETL 流程...")

# ==========================================
# 階段一：讀取 11 棟建築電力資料
# ==========================================
df_list = []
for file in excel_files:
    filename = os.path.basename(file)
    if filename.startswith('~$'): continue
    try:
        temp_df = pd.read_excel(file, skiprows=1)
        temp_df = temp_df.iloc[:, [0, 1]].copy()
        temp_df.columns = ['Time', 'kW']
        
        building_name = filename.replace('.xlsx', '')
        if '共同2025' in building_name: building_name = '共同教學館'
        temp_df['Building'] = building_name
        temp_df['kW'] = pd.to_numeric(temp_df['kW'], errors='coerce')
        df_list.append(temp_df)
    except Exception as e:
        print(f"讀取 Excel 失敗 {filename}: {e}")

raw_data = pd.concat(df_list, ignore_index=True)
raw_data['Time'] = pd.to_datetime(raw_data['Time'], errors='coerce')
raw_data = raw_data.dropna(subset=['Time']).sort_values(by=['Building', 'Time'])
raw_data['kW'] = raw_data.groupby('Building')['kW'].ffill().bfill()

# ==========================================
# 階段二：讀取 CODiS 氣象資料
# ==========================================
weather_records = []
for file in sorted(csv_files):
    filename = os.path.basename(file)
    if 'DataState' in filename: continue
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('466920'):
                    cols = line.split(',')
                    if len(cols) > 5:
                        weather_records.append([cols[1].strip(), cols[4].strip()])
    except Exception as e:
        print(f"讀取 CSV 失敗 {filename}: {e}")

weather_df = pd.DataFrame(weather_records, columns=['TimeStr', 'Temperature'])

def parse_cwa_time(t_str):
    t_str = str(t_str)
    if t_str.endswith('24'):
        return pd.to_datetime(t_str[:-2] + '00', format='%Y%m%d%H') + pd.Timedelta(days=1)
    return pd.to_datetime(t_str, format='%Y%m%d%H', errors='coerce')

weather_df['Time'] = weather_df['TimeStr'].apply(parse_cwa_time)
weather_df['Temperature'] = pd.to_numeric(weather_df['Temperature'], errors='coerce')
weather_df = weather_df.dropna(subset=['Time']).drop_duplicates(subset=['Time'])[['Time', 'Temperature']]

# ==========================================
# 階段三：特徵合併與清洗
# ==========================================
final_vpp_data = pd.merge(raw_data, weather_df, on='Time', how='left')

# 時間特徵
final_vpp_data['Hour'] = final_vpp_data['Time'].dt.hour
final_vpp_data['Month'] = final_vpp_data['Time'].dt.month
final_vpp_data['Is_Weekend'] = final_vpp_data['Time'].dt.dayofweek.isin([5, 6]).astype(int)

clean_vpp_data = final_vpp_data.copy()

# 氣象清洗
clean_vpp_data.loc[(clean_vpp_data['Temperature'] <= 0) | (clean_vpp_data['Temperature'] >= 45), 'Temperature'] = pd.NA
clean_vpp_data['Temperature'] = clean_vpp_data.groupby('Building')['Temperature'].transform(
    lambda x: x.interpolate(method='linear').ffill().bfill()
)

# 電力清洗
clean_vpp_data = clean_vpp_data[clean_vpp_data['kW'] > 0]
def filter_power_spikes(group):
    q1 = group['kW'].quantile(0.25)
    q3 = group['kW'].quantile(0.75)
    iqr = q3 - q1
    return group[group['kW'] <= (q3 + 1.5 * iqr)]
clean_vpp_data = clean_vpp_data.groupby('Building', group_keys=False).apply(filter_power_spikes)

# ==========================================
# 階段四：存檔 (建立檢查點)
# ==========================================
output_path = './clean_vpp_data.csv'
clean_vpp_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ ETL 流程完成！乾淨資料已儲存至: {output_path}")