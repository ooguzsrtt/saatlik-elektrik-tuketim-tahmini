# saatlik-elektrik-tuketim-tahmini
⚡ Saatlik Elektrik Tüketim Tahmini (LightGBM + ERA5)

Bu proje, Türkiye’nin saatlik elektrik tüketimini ERA5 hava durumu verileri, takvim özellikleri ve gelişmiş zaman serisi feature engineering teknikleriyle tahmin eder. Model olarak LightGBM kullanılmıştır.

📌 Amaç

Saatlik elektrik tüketimini yüksek doğrulukla tahmin etmek

ERA5 hava verilerini otomatik çekmek

Lag, rolling, cyclic encoding gibi özellik mühendisliği adımlarını uygulamak

LightGBM ile düşük hata oranına sahip bir model geliştirmek

🧠 Kullanılan Yöntem
LightGBM Regressor

Hızlı ve kararlı

Zaman serisi veri setlerinde yüksek başarı

Optuna ile hiperparametre optimizasyonu

Feature Engineering

ERA5 sıcaklık ve hissedilen sıcaklık ortalamaları

HDD & CDD

Mevsim ve zaman dilimi kategorileri

1–720 saat arası lag değişkenleri

3–720 saat arası rolling mean

Sine/Cosine cyclic hour encoding

