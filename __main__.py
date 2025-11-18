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
df_raw = pd.read_excel("1.xlsx")
df_ready, lag_roll_cols = prepare_energy_dataset(df_raw)

# ---------------------------------------------------
# TEST / VALIDATION / TRAIN SPLIT
# ---------------------------------------------------
# Test: 01.01.2025 (tüm günü) ve 17.11.2025 (tüm günü)
test_mask = (df_ready["Tarih"].dt.date == pd.to_datetime("2025-01-01").date())

test = df_ready[test_mask].copy()
remaining = df_ready[~test_mask].copy()

# Kalan verinin %20'si validation (zaman sırasına göre son %20)
val_size = int(len(remaining) * 0.20)
val = remaining.iloc[-val_size:].copy()
train = remaining.iloc[:-val_size].copy()

print(f"Train boyutu: {len(train)}")
print(f"Validation boyutu: {len(val)}")
print(f"Test boyutu: {len(test)}")
print(f"\nTest tarihleri:")
print(test["Tarih"].dt.date.unique())

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

X_train = train[feature_cols]
y_train = train["Tüketim Miktarı(MWh)"]

X_val = val[feature_cols]
y_val = val["Tüketim Miktarı(MWh)"]

X_test = test[feature_cols]
y_test = test["Tüketim Miktarı(MWh)"]

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

# Feature importance
importances = lgb_model.feature_importances_
idx = np.argsort(importances)[-20:]  # En önemli 20 feature

plt.figure(figsize=(10, 8))
plt.barh(np.array(feature_cols)[idx], importances[idx])
plt.title("LightGBM Feature Importance (Top 20)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Test tahminlerini görselleştir
plt.figure(figsize=(15, 6))
plt.plot(test["Tarih"], y_test.values, label="Gerçek", marker='o', linewidth=2)
plt.plot(test["Tarih"], lgb_pred_test, label="Tahmin", marker='x', linewidth=2)
plt.xlabel("Tarih")
plt.ylabel("Tüketim (MWh)")
plt.title("Test Seti: Gerçek vs Tahmin")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
