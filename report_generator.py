import pandas as pd
import joblib

def main():
    print("="*60)
    print(" 🏎️  F1 UNDERCUT YAPAY ZEKA STRATEJİ RAPORU ÜRETİCİ 🏎️ ")
    print("="*60)

    # 1. Model ve Veriyi Yükle
    try:
        model = joblib.load('f1_undercut_model.pkl')
        df = pd.read_csv('f1_22_25_undercut_data.csv')
    except FileNotFoundError:
        print("HATA: Model veya CSV dosyası bulunamadı!")
        return

    # 2. Feature + Tahmin
    features = ['Gap', 'Driver_TyreLife', 'Ahead_TyreLife', 'Tyre_Advantage']
    X = df[features]
    df['AI_Probability_%'] = (model.predict_proba(X)[:, 1] * 100).round(2)

    # 3. En iyi senaryolar
    top_undercuts = df.sort_values(by='AI_Probability_%', ascending=False).head(5)
    real_successes = df[df['Success'] == 1].head(5)

    # 4. FEATURE IMPORTANCE (🔥 kritik upgrade)
    importance = model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance (%)': (importance * 100)
    }).sort_values(by='Importance (%)', ascending=False)

    most_important = feature_df.iloc[0]['Feature']

    print("\n[BÖLÜM 1: EN YÜKSEK POTANSİYELLİ DURUMLAR]")
    cols = ['Race', 'Driver', 'Car_Ahead', 'Gap', 'Tyre_Advantage', 'AI_Probability_%', 'Success']
    print(top_undercuts[cols].to_string(index=False))

    print("\n[BÖLÜM 2: GERÇEK BAŞARILI UNDERCUT'LAR]")
    print(real_successes[cols].to_string(index=False))

    print("\n[BÖLÜM 3: STRATEJİK ÖZET]")
    avg_gap_success = df[df['Success'] == 1]['Gap'].mean()
    avg_gap_fail = df[df['Success'] == 0]['Gap'].mean()

    print(f"• Başarılı ortalama gap: {avg_gap_success:.3f} sn")
    print(f"• Başarısız ortalama gap: {avg_gap_fail:.3f} sn")
    print(f"• En kritik değişken: {most_important}")
    print(f"• %40 üzeri senaryo sayısı: {len(df[df['AI_Probability_%'] > 40])}")

    # 🔥 YENİ: Otomatik yorum üretimi
    yorum = ""
    if most_important == "Tyre_Advantage":
        yorum = "Undercut başarısında lastik avantajı belirleyici faktördür."
    elif most_important == "Ahead_TyreLife":
        yorum = "Rakibin lastik aşınması en kritik fırsat unsurudur."
    elif most_important == "Gap":
        yorum = "Saniye farkı undercut başarısını doğrudan etkiler."
    else:
        yorum = "Pilot lastik durumu belirleyici rol oynar."

    print(f"• AI Yorumu: {yorum}")

    # 5. Markdown Rapor
    report_filename = "AI_Strategy_Report.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("# F1 Undercut Strateji Analiz Raporu\n\n")

        f.write("## En Yüksek Olasılıklı Senaryolar\n")
        f.write(top_undercuts[cols].to_markdown(index=False))

        f.write("\n\n## Gerçek Başarılı Örnekler\n")
        f.write(real_successes[cols].to_markdown(index=False))

        f.write("\n\n## Feature Importance\n")
        f.write(feature_df.to_markdown(index=False))

        f.write("\n\n## Stratejik Bulgular\n")
        f.write(f"- Başarılı undercut ortalama gap: {avg_gap_success:.3f} sn\n")
        f.write(f"- Başarısız undercut ortalama gap: {avg_gap_fail:.3f} sn\n")
        f.write(f"- En önemli değişken: {most_important}\n")
        f.write(f"- Yorum: {yorum}\n")

    print(f"\n✅ Rapor oluşturuldu: {report_filename}")

if __name__ == "__main__":
    main()