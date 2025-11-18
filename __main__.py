import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMRegressor

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def prepare_energy_dataset(df, start_date="2024-01-01", end_date="2025-01-01"):
    import pandas as pd
    import numpy as np
    import requests

    # ---------------------------------------------------
    # 0) ERA5 SICAKLIK & HISSEDILEN SICAKLIK (Otamatik)
    # ---------------------------------------------------
    points = [
        (42.0, 36.0),  
        (36.0, 33.0),  
        (38.5, 26.5),  
        (39.5, 44.0),  
        (39.0, 35.0)   
    ]

    def get_temp(lat, lon):
        url = (
            "https://archive-api.open-meteo.com/v1/era5?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            "&hourly=temperature_2m,apparent_temperature"
            "&timezone=Europe/Istanbul"
        )
        r = requests.get(url).json()
        df_t = pd.DataFrame({
            "Tarih": r["hourly"]["time"],
            f"T_{lat}_{lon}": r["hourly"]["temperature_2m"],
            f"A_{lat}_{lon}": r["hourly"]["apparent_temperature"]
        })
        df_t["Tarih"] = pd.to_datetime(df_t["Tarih"])
        return df_t

    temp_frames = [get_temp(lat, lon) for lat, lon in points]
    df_temp = temp_frames[0]
    for f in temp_frames[1:]:
        df_temp = df_temp.merge(f, on="Tarih", how="left")

    temp_cols = [c for c in df_temp.columns if c.startswith("T_")]
    app_cols  = [c for c in df_temp.columns if c.startswith("A_")]

    df_temp["Sicaklik_Ortalama"] = df_temp[temp_cols].mean(axis=1)
    df_temp["Hissedilen_Sicaklik_Ortalama"] = df_temp[app_cols].mean(axis=1)

    df_temp = df_temp[["Tarih", "Sicaklik_Ortalama", "Hissedilen_Sicaklik_Ortalama"]]

    # ---------------------------------------------------
    # 1) Merge Ana Dataset + Temp
    # ---------------------------------------------------
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.merge(df_temp, on="Tarih", how="left")

    # ---------------------------------------------------
    # 2) Holiday & Calendar
    # ---------------------------------------------------
    turkey_holidays = [
        "2024-01-01", "2024-04-09", "2024-04-10", "2024-04-11", "2024-04-12",
        "2024-04-23", "2024-05-01", "2024-05-19", "2024-06-15",
        "2024-06-16", "2024-06-17", "2024-06-18", "2024-06-19",
        "2024-07-15", "2024-08-30", "2024-10-28", "2024-10-29", "2024-12-31"
    ]
    holiday_dates = pd.to_datetime(turkey_holidays)
    df["is_holiday"] = df["Tarih"].dt.date.isin(holiday_dates.date).astype(int)

    df["month"] = df["Tarih"].dt.month
    df["dayofyear"] = df["Tarih"].dt.dayofyear
    df["dayofweek"] = df["Tarih"].dt.dayofweek
    df["hour"] = df["Tarih"].dt.hour
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # ---------------------------------------------------
    # 3) Zaman Dilimi
    # ---------------------------------------------------
    def zaman(h):
        if 6 <= h <= 10: return "sabah"
        if 11 <= h <= 16: return "oglen"
        if 17 <= h <= 23: return "aksam"
        return "gece"

    df["zaman"] = df["hour"].apply(zaman)

    # ---------------------------------------------------
    # 4) Mevsim
    # ---------------------------------------------------
    def season(m):
        if m in [12,1,2]: return "Kis"
        if m in [3,4,5]: return "Ilkbahar"
        if m in [6,7,8]: return "Yaz"
        return "Sonbahar"

    df["Mevsim"] = df["month"].apply(season)

    # ---------------------------------------------------
    # 5) Lag & Rolling
    # ---------------------------------------------------
    lag_list = [1,2,3,6,12,24,48,72,96,120,144,168,336,504,720]
    roll_list = [3,6,12,24,48,72,168,336,720]

    for l in lag_list:
        df[f"lag_{l}"] = df["Tüketim Miktarı(MWh)"].shift(l)

    for r in roll_list:
        df[f"roll_{r}"] = df["Tüketim Miktarı(MWh)"].rolling(r).mean()

    # ---------------------------------------------------
    # 6) Sıcaklık Türevi
    # ---------------------------------------------------
    df["Temp_Sq"] = df["Sicaklik_Ortalama"] ** 2
    df["HDD"] = (18 - df["Sicaklik_Ortalama"]).clip(lower=0)
    df["CDD"] = (df["Sicaklik_Ortalama"] - 18).clip(lower=0)
    df["FeelsLike"] = df["Hissedilen_Sicaklik_Ortalama"]

    # ---------------------------------------------------
    # 7) Dummy Kolonları
    # ---------------------------------------------------
    df = pd.get_dummies(df, columns=["zaman", "Mevsim"], drop_first=True)

    # Dummy garanti
    for col in ["zaman_sabah","zaman_oglen","zaman_aksam"]:
        if col not in df.columns: df[col] = 0
    for col in ["Mevsim_Ilkbahar","Mevsim_Yaz","Mevsim_Sonbahar"]:
        if col not in df.columns: df[col] = 0

    # ---------------------------------------------------
    # 8) Leakage Önleme → NA'ları at
    # ---------------------------------------------------
    lag_roll_cols = [f"lag_{l}" for l in lag_list] + [f"roll_{r}" for r in roll_list]
    df = df.dropna(subset=lag_roll_cols).copy()

    # Lag ve roll kolonlarını return et
    return df, lag_roll_cols


# ---------------------------------------------------
# VERİ HAZIRLAMA
# ---------------------------------------------------
df_raw = pd.read_excel("Gercek_Zamanli_Tuketim-17102025-17112025.xlsx")
df_ready, lag_roll_cols = prepare_energy_dataset(df_raw)

# ---------------------------------------------------
# TEST / VALIDATION / TRAIN SPLIT
# ---------------------------------------------------
# Test: 01.01.2025 (tüm günü) ve 17.11.2025 (tüm günü)
test_mask = (df_ready["Tarih"].dt.date == pd.to_datetime("2025-11-17").date())

test_df = df_ready[test_mask].copy()
remaining_df = df_ready[~test_mask].copy()

# Kalan verinin %20'si validation (zaman sırasına göre son %20)
val_size = int(len(remaining_df) * 0.20)
val_df = remaining_df.iloc[-val_size:].copy()
train_df = remaining_df.iloc[:-val_size].copy()

print(f"Train boyutu: {len(train_df)}")
print(f"Validation boyutu: {len(val_df)}")
print(f"Test boyutu: {len(test_df)}")
print(f"\nTest tarihleri:")
print(test_df["Tarih"].dt.date.unique())

# ---------------------------------------------------
# FEATURE SET
# ---------------------------------------------------
base_cols = [
    "Sicaklik_Ortalama","Hissedilen_Sicaklik_Ortalama","FeelsLike",
    "Temp_Sq","HDD","CDD",
    "month","dayofyear","dayofweek",
    "is_weekend","is_holiday",
    "hour_sin","hour_cos",
    "zaman_sabah","zaman_oglen","zaman_aksam",
    "Mevsim_Ilkbahar","Mevsim_Yaz","Mevsim_Sonbahar"
]

feature_cols = base_cols + lag_roll_cols

X_train = train_df[feature_cols]
y_train = train_df["Tüketim Miktarı(MWh)"]

X_val = val_df[feature_cols]
y_val = val_df["Tüketim Miktarı(MWh)"]

X_test = test_df[feature_cols]
y_test = test_df["Tüketim Miktarı(MWh)"]

# ---------------------------------------------------
# MODEL EĞİTİMİ
# ---------------------------------------------------
lgb_model = LGBMRegressor(
    n_estimators=2356,
    learning_rate=0.023759508814709875,
    max_depth=11,
    num_leaves=283,
    subsample=0.9579103897143751,
    colsample_bytree=0.9718295547292943,
    reg_lambda=0.30049013952702686,
    reg_alpha=0.09141699945985104,
    min_child_samples=11,
    random_state=42,
    verbosity=-1
)

# LightGBM'de early stopping callback ile
callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False)
]

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l1",
    callbacks=callbacks
)

# ---------------------------------------------------
# TAHMİNLER VE METRİKLER
# ---------------------------------------------------
lgb_pred_train = lgb_model.predict(X_train)
lgb_pred_val   = lgb_model.predict(X_val)
lgb_pred_test  = lgb_model.predict(X_test)

print("\n" + "="*50)
print("--- LIGHTGBM TRAIN METRİKLERİ ---")
print("="*50)
train_metrics = metrics(y_train, lgb_pred_train)
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n" + "="*50)
print("--- LIGHTGBM VALIDATION METRİKLERİ ---")
print("="*50)
val_metrics = metrics(y_val, lgb_pred_val)
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n" + "="*50)
print("--- LIGHTGBM TEST METRİKLERİ ---")
print("="*50)
test_metrics = metrics(y_test, lgb_pred_test)
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# Test günlerinin detaylı sonuçları
print("\n" + "="*50)
print("TEST GÜNLERİ DETAYLI SONUÇLAR")
print("="*50)
test_results = test[["Tarih"]].copy()
test_results["Gerçek"] = y_test.values
test_results["Tahmin"] = lgb_pred_test
test_results["Hata"] = test_results["Gerçek"] - test_results["Tahmin"]
test_results["MAPE"] = np.abs(test_results["Hata"] / test_results["Gerçek"]) * 100

for date in test_results["Tarih"].dt.date.unique():
    print(f"\n{date}:")
    day_data = test_results[test_results["Tarih"].dt.date == date]
    print(f"  Ortalama MAPE: {day_data['MAPE'].mean():.2f}%")
    print(f"  Maksimum MAPE: {day_data['MAPE'].max():.2f}%")
    print(f"  Minimum MAPE: {day_data['MAPE'].min():.2f}%")



# ============================================
# TRAIN / VAL / TEST + PREDICT EXCEL EXPORT
# ============================================

# Train dataframe + predictions
train_out = train_df.copy()
train_out["Predict"] = lgb_pred_train

# Validation dataframe + predictions
val_out = val_df.copy()
val_out["Predict"] = lgb_pred_val

# Test dataframe + predictions
test_out = test_df.copy()
test_out["Predict"] = lgb_pred_test

# Hepsini tek Excel'de farklı sheet'lere kaydet
with pd.ExcelWriter("train_val_test_predictions.xlsx") as writer:
    train_out.to_excel(writer, sheet_name="TRAIN", index=False)
    val_out.to_excel(writer, sheet_name="VAL", index=False)
    test_out.to_excel(writer, sheet_name="TEST", index=False)

print("Excel başarıyla oluşturuldu: train_val_test_predictions.xlsx")



# =============================================================================
# GÖRSELLEŞTIRME (AYRI AYRI PLOTLAR)
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')

# Hazırlık
test_results = test_df[["Tarih"]].copy()
test_results["MAPE"] = np.abs((y_test.values - lgb_pred_test) / y_test.values) * 100
test_results["Gün"] = test_results["Tarih"].dt.date
test_results["Saat"] = test_results["Tarih"].dt.hour
residuals_test = y_test.values - lgb_pred_test

print("\n" + "="*70)
print("GÖRSELLEŞTIRME BAŞLIYOR...")
print("="*70)

# 1. FEATURE IMPORTANCE (TOP 25)
print("\n📊 Grafik 1/12: Özellik Önem Sıralaması")
plt.figure(figsize=(12, 8))
importances = lgb_model.feature_importances_
idx = np.argsort(importances)[-25:]
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
plt.barh(np.array(feature_cols)[idx], importances[idx], color=colors)
plt.title("En Önemli 25 Özellik", fontsize=16, fontweight='bold')
plt.xlabel("Önem Skoru", fontsize=12)
plt.tight_layout()
plt.show()

# 2. TEST SETİ: GERÇEK VS TAHMİN (ZAMAN SERİSİ)
print("\n📊 Grafik 2/12: Test Seti Zaman Serisi")
plt.figure(figsize=(16, 6))
plt.plot(test_df["Tarih"], y_test.values, label="Gerçek", marker='o', linewidth=2.5, markersize=6, color='#2E86AB')
plt.plot(test_df["Tarih"], lgb_pred_test, label="Tahmin", marker='x', linewidth=2.5, markersize=6, color='#A23B72')
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Tüketim (MWh)", fontsize=12)
plt.title("Test Seti: Gerçek Değerler ve Tahminler", fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. SCATTER PLOT: GERÇEK VS TAHMİN (TÜM SETLER)
print("\n📊 Grafik 3/12: Dağılım Grafiği (Tüm Setler)")
plt.figure(figsize=(10, 10))
plt.scatter(y_train, lgb_pred_train, alpha=0.3, s=10, label='Eğitim', color='#3A86FF')
plt.scatter(y_val, lgb_pred_val, alpha=0.5, s=20, label='Doğrulama', color='#FB5607')
plt.scatter(y_test, lgb_pred_test, alpha=0.8, s=40, label='Test', color='#FF006E', edgecolors='black', linewidth=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='İdeal Çizgi')
plt.xlabel("Gerçek Değer (MWh)", fontsize=12)
plt.ylabel("Tahmin Değeri (MWh)", fontsize=12)
plt.title("Gerçek ve Tahmin Değerleri Karşılaştırması", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. RESIDUAL PLOT (TEST)
print("\n📊 Grafik 4/12: Hata Analizi Grafiği")
plt.figure(figsize=(12, 6))
plt.scatter(lgb_pred_test, residuals_test, alpha=0.6, s=40, color='#8338EC', edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Sıfır Hata Çizgisi')
plt.xlabel("Tahmin Değeri (MWh)", fontsize=12)
plt.ylabel("Hata (Gerçek - Tahmin)", fontsize=12)
plt.title("Artık Hata Grafiği (Test Seti)", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. TEST GÜNLERİNE GÖRE MAPE DAĞILIMI
print("\n📊 Grafik 5/12: Günlük MAPE Değerleri")
plt.figure(figsize=(10, 6))
mape_by_day = test_results.groupby("Gün")["MAPE"].mean()
colors_mape = ['#06D6A0' if x < 3 else '#FFD60A' if x < 5 else '#EF476F' for x in mape_by_day.values]
bars = plt.bar(range(len(mape_by_day)), mape_by_day.values, color=colors_mape, edgecolor='black', linewidth=1.5)
plt.axhline(y=3, color='green', linestyle='--', linewidth=2, label='Hedef (%3)')
plt.xticks(range(len(mape_by_day)), [str(d) for d in mape_by_day.index], rotation=45, ha='right')
plt.ylabel("Ortalama MAPE (%)", fontsize=12)
plt.title("Günlük Ortalama Mutlak Yüzde Hatası", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

for i, (bar, val) in enumerate(zip(bars, mape_by_day.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# 6. SAATLIK MAPE DAĞILIMI (TEST)
print("\n📊 Grafik 6/12: Saatlik MAPE Dağılımı")
plt.figure(figsize=(14, 6))
mape_by_hour = test_results.groupby("Saat")["MAPE"].mean()
plt.plot(mape_by_hour.index, mape_by_hour.values, marker='o', linewidth=2.5, markersize=8, color='#F72585')
plt.axhline(y=3, color='green', linestyle='--', linewidth=2, label='Hedef (%3)')
plt.xlabel("Saat", fontsize=12)
plt.ylabel("Ortalama MAPE (%)", fontsize=12)
plt.title("Saatlik Ortalama Mutlak Yüzde Hatası (Test)", fontsize=16, fontweight='bold')
plt.xticks(range(0, 24, 2))
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. HATA DAĞILIMI (HİSTOGRAM - TEST)
print("\n📊 Grafik 7/12: Hata Dağılımı Histogramı")
plt.figure(figsize=(10, 6))
plt.hist(residuals_test, bins=30, alpha=0.7, color='#4CC9F0', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Sıfır Hata')
plt.xlabel("Hata Değeri (MWh)", fontsize=12)
plt.ylabel("Frekans", fontsize=12)
plt.title("Test Seti Hata Dağılımı", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 8. MUTLAK HATA YÜZDESI DAĞILIMI (TEST)
print("\n📊 Grafik 8/12: MAPE Dağılımı Histogramı")
plt.figure(figsize=(10, 6))
plt.hist(test_results["MAPE"], bins=30, alpha=0.7, color='#7209B7', edgecolor='black')
plt.axvline(x=3, color='green', linestyle='--', linewidth=2, label='Hedef (%3)')
plt.xlabel("MAPE Değeri (%)", fontsize=12)
plt.ylabel("Frekans", fontsize=12)
plt.title("Test Seti MAPE Dağılımı", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 9. VALIDATION SETİ: GERÇEK VS TAHMİN (SON 7 GÜN)
print("\n📊 Grafik 9/12: Doğrulama Seti Son 7 Gün")
plt.figure(figsize=(16, 6))
# Son 168 saat (7 gün) validation verisini al
val_rows = min(168, len(val_df))
val_last_week = val_df.tail(val_rows).copy()
y_val_last = val_last_week["Tüketim Miktarı(MWh)"]
X_val_last = val_last_week[feature_cols]
pred_val_last = lgb_model.predict(X_val_last)
plt.plot(val_last_week["Tarih"], y_val_last.values, label="Gerçek", linewidth=2, alpha=0.8, color='#06D6A0')
plt.plot(val_last_week["Tarih"], pred_val_last, label="Tahmin", linewidth=2, alpha=0.8, color='#EF476F')
plt.fill_between(val_last_week["Tarih"], y_val_last.values, pred_val_last, alpha=0.2, color='gray')
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Tüketim (MWh)", fontsize=12)
plt.title(f"Doğrulama Seti (Son {val_rows//24} Gün): Gerçek ve Tahmin", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 10. SICAKLIK VS TÜKETİM İLİŞKİSİ (TEST)
print("\n📊 Grafik 10/12: Sıcaklık ve Tüketim İlişkisi")
plt.figure(figsize=(10, 8))
test_temp = test_df[["Sicaklik_Ortalama"]].values
scatter = plt.scatter(test_temp, y_test.values, c=test_df["hour"].values, cmap='twilight', 
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
plt.xlabel("Sıcaklık (°C)", fontsize=12)
plt.ylabel("Tüketim (MWh)", fontsize=12)
plt.title("Sıcaklık ve Enerji Tüketimi İlişkisi (Saate Göre Renklendirilmiş)", fontsize=16, fontweight='bold')
cbar = plt.colorbar(scatter)
cbar.set_label('Saat', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 11. METRİK KARŞILAŞTIRMASI (BAR CHART)
print("\n📊 Grafik 11/12: Performans Metrikleri Karşılaştırması")
plt.figure(figsize=(12, 6))
metrics_names = ['MAE', 'RMSE', 'MAPE']
train_vals = [train_metrics['MAE'], train_metrics['RMSE'], train_metrics['MAPE']]
val_vals = [val_metrics['MAE'], val_metrics['RMSE'], val_metrics['MAPE']]
test_vals = [test_metrics['MAE'], test_metrics['RMSE'], test_metrics['MAPE']]

x = np.arange(len(metrics_names))
width = 0.25

plt.bar(x - width, train_vals, width, label='Eğitim', color='#3A86FF', edgecolor='black')
plt.bar(x, val_vals, width, label='Doğrulama', color='#FB5607', edgecolor='black')
plt.bar(x + width, test_vals, width, label='Test', color='#FF006E', edgecolor='black')

plt.ylabel('Metrik Değeri', fontsize=12)
plt.title('Model Performans Metrikleri (Eğitim/Doğrulama/Test)', fontsize=16, fontweight='bold')
plt.xticks(x, metrics_names)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 12. TEST GÜNLERİ DETAYLI (HER GÜN AYRI)
print("\n📊 Grafik 12/12: Test Günleri Detaylı Analiz")
plt.figure(figsize=(14, 7))
for date in test_df["Tarih"].dt.date.unique():
    day_mask = test_df["Tarih"].dt.date == date
    day_data = test_df[day_mask]
    day_true = y_test[test_df["Tarih"].dt.date == date].values
    day_pred = lgb_pred_test[test_df["Tarih"].dt.date == date]
    
    hours = day_data["Tarih"].dt.hour
    plt.plot(hours, day_true, marker='o', linewidth=2.5, markersize=7, label=f'{date} - Gerçek', alpha=0.8)
    plt.plot(hours, day_pred, marker='x', linewidth=2.5, markersize=7, label=f'{date} - Tahmin', alpha=0.8, linestyle='--')

plt.xlabel("Saat", fontsize=12)
plt.ylabel("Tüketim (MWh)", fontsize=12)
plt.title("Test Günleri Saatlik Karşılaştırma", fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()
