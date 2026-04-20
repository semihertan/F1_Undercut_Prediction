import fastf1
import pandas as pd

# Cache dizinini belirtiyoruz
fastf1.Cache.enable_cache('f1_cache') 

def get_sample_race():
    print("🏎️ Örnek yarış verisi hazırlanıyor (Zandvoort 2023)...")
    
    # Stratejik açıdan zengin bir yarış
    session = fastf1.get_session(2023, 'Zandvoort', 'R')
    session.load(telemetry=False, weather=False)
    
    laps = session.laps
    
    # Raporda şık durması için sadece kritik sütunlar
    cols_to_show = [
        'Driver', 'LapNumber', 'Stint', 'Compound', 
        'TyreLife', 'Position', 'LapTime', 'PitOutTime'
    ]
    
    # Örnek olarak Verstappen ve Alonso'nun ilk turları
    sample_df = laps[laps['Driver'].isin(['VER', 'ALO'])].head(15).copy()
    sample_df = sample_df[cols_to_show]
    
    # HATALI KISIM BURADA DÜZELTİLDİ:
    # PitOutTime sütunu boş değilse (notnull) True döner.
    sample_df['Is_Pit_Stop'] = sample_df['PitOutTime'].notnull()
    
    # Zaman formatını saniyeye çevirip yuvarlayalım (Raporda temiz görünür)
    sample_df['LapTime'] = sample_df['LapTime'].dt.total_seconds().round(3)
    
    print("\n🔹 RAPOR İÇİN ÖRNEK HAM VERİ TABLOSU (İLK 15 SATIR):")
    print("-" * 90)
    print(sample_df.to_string(index=False))
    print("-" * 90)
    
    # Dosyaya kaydet
    sample_df.to_csv('sample_race_snapshot.csv', index=False)
    print("\n✅ 'sample_race_snapshot.csv' olarak kaydedildi.")

if __name__ == "__main__":
    get_sample_race()