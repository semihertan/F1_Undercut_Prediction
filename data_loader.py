import fastf1 as ff1
import pandas as pd
import os

def main():
    print("TÜM 2021-2025 SEZONLARI ÇEKİLİYOR...")

    # 1. Önbellek (Cache) Ayarı
    cache_dir = 'f1_cache'
    os.makedirs(cache_dir, exist_ok=True)
    ff1.Cache.enable_cache(cache_dir)

    master_dataset = []

    # Sadece yeni regülasyon dönemini alıyoruz!
    for year in [2022, 2023, 2024, 2025]:
        print(f"\n{'='*40}")
        print(f"       {year} SEZONU BAŞLIYOR")
        print(f"{'='*40}")
        
        for round_num in range(1, 26):
            try:
                event = ff1.get_event(year, round_num)
                gp_name = event.EventName
                
                print(f"---> {year} Raund {round_num}: {gp_name} İşleniyor <---")
                
                session = ff1.get_session(year, round_num, 'R')
                session.load(telemetry=False, weather=False)
                
                df = session.laps[['DriverNumber', 'LapNumber', 'LapTime', 'PitInTime', 
                                   'Compound', 'TyreLife', 'TrackStatus', 'Time', 'Position']].copy()
                
                df = df.dropna(subset=['LapTime', 'Position', 'DriverNumber'])
                df['Total_Time_Sec'] = df['Time'].dt.total_seconds()
                
                invalid_status = ['4', '5', '6', '7']
                df_clean = df[~df['TrackStatus'].astype(str).apply(lambda x: any(s in x for s in invalid_status))].copy()
                pits_df = df_clean[df_clean['PitInTime'].notnull()].copy()
                
                # X ve Y Özelliklerini Hesaplama
                for index, pit in pits_df.iterrows():
                    driver = pit['DriverNumber']
                    pit_lap = pit['LapNumber']
                    prev_lap_num = pit_lap - 1
                    
                    if prev_lap_num <= 0: continue
                        
                    prev_lap_data = df_clean[df_clean['LapNumber'] == prev_lap_num]
                    driver_row = prev_lap_data[prev_lap_data['DriverNumber'] == driver]
                    
                    if driver_row.empty: continue
                    driver_pos = driver_row['Position'].values[0]
                    if pd.isna(driver_pos) or driver_pos <= 1.0: continue
                        
                    ahead_row = prev_lap_data[prev_lap_data['Position'] == (driver_pos - 1.0)]
                    if ahead_row.empty: continue
                        
                    gap = driver_row['Total_Time_Sec'].values[0] - ahead_row['Total_Time_Sec'].values[0]
                    car_ahead = ahead_row['DriverNumber'].values[0]
                    
                    ahead_pits = pits_df[(pits_df['DriverNumber'] == car_ahead) & (pits_df['LapNumber'] > pit_lap)]
                    if ahead_pits.empty: continue
                        
                    ahead_pit_lap = ahead_pits.iloc[0]['LapNumber']
                    
                    if (ahead_pit_lap - pit_lap) > 5: continue
                        
                    compare_lap = ahead_pit_lap
                    lap_data = df_clean[df_clean['LapNumber'] == compare_lap]
                    
                    drv_t = lap_data[lap_data['DriverNumber'] == driver]
                    ahd_t = lap_data[lap_data['DriverNumber'] == car_ahead]
                    
                    if drv_t.empty or ahd_t.empty: continue
                        
                    success = 1 if drv_t['Total_Time_Sec'].values[0] < ahd_t['Total_Time_Sec'].values[0] else 0
                    
                    master_dataset.append({
                        'Race': gp_name,
                        'Driver': driver,
                        'Car_Ahead': car_ahead,
                        'Gap': round(gap, 3), 
                        'Driver_TyreLife': driver_row['TyreLife'].values[0], 
                        'Ahead_TyreLife': ahead_row['TyreLife'].values[0],   
                        'Tyre_Advantage': ahead_row['TyreLife'].values[0] - driver_row['TyreLife'].values[0],
                        'Success': success
                    })
                    
            except Exception as e:
                print(f"Hata/İptal: {year} Raund {round_num} atlandı. Detay: {e}")

    # Döngü bitti, DataFrame oluştur
    final_ml_df = pd.DataFrame(master_dataset)

    print(f"\n=== VERİ TOPLAMA BİTTİ! TOPLAM {len(final_ml_df)} ADET UNDERCUT SAVAŞI BULUNDU ===")
    
    # Bilgisayara CSV olarak kaydetme işlemi
    csv_filename = 'f1_22_25_undercut_data.csv'
    final_ml_df.to_csv(csv_filename, index=False)
    print(f"Tüm veriler '{csv_filename}' dosyasına başarıyla kaydedildi!")

if __name__ == "__main__":
    main()