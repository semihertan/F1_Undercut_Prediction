import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE

def main():
    print("="*55)
    print("🏎️  F1 STRATEJİ MODELİ EĞİTİM PANELİ (SMOTE DESTEKLİ)")
    print("="*55 + "\n")

    # 1. Veriyi Yükle
    try:
        df = pd.read_csv('f1_22_25_undercut_data.csv')
    except FileNotFoundError:
        print("HATA: 'f1_22_25_undercut_data.csv' bulunamadı.")
        return

    df = df.dropna()

    # 2. X ve y Ayırımı
    X = df[['Gap', 'Driver_TyreLife', 'Ahead_TyreLife', 'Tyre_Advantage']]
    y = df['Success']

    # 3. Veriyi Böl (%80 Eğitim, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. SMOTE UYGULAMA (Sadece eğitim setine)
    # Bu adım, 22 olan başarılı örnek sayısını sentetik olarak artırır
    print("🪄 SMOTE ile veri dengeleniyor...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"📊 Veri Özeti:")
    print(f"- Orijinal Başarılı Sayısı (Eğitim): {sum(y_train)}")
    print(f"- SMOTE Sonrası Başarılı Sayısı   : {sum(y_train_resampled)}")
    print(f"- Test Seti Örnek Sayısı          : {len(X_test)}\n")

    # 5. Modelleri Tanımla
    # Veri SMOTE ile dengelendiği için class_weight kullanmaya gerek kalmadı
    log_reg = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 6. Modelleri Eğit (Dengelenmiş veriyle)
    print("🚀 Modeller eğitiliyor, lütfen bekleyin...")
    log_reg.fit(X_train_resampled, y_train_resampled)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # 7. Tahminler
    log_probs = log_reg.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    # 8. SONUÇLAR
    print("\n" + "="*45)
    print("      [1] LOGISTIC REGRESSION (SMOTE)")
    print("="*45)
    print(f"ROC-AUC Skoru: {roc_auc_score(y_test, log_probs):.3f}")
    print(classification_report(y_test, log_reg.predict(X_test), zero_division=0))

    print("\n" + "="*45)
    print("      [2] RANDOM FOREST (SMOTE)")
    print("="*45)
    print(f"ROC-AUC Skoru: {roc_auc_score(y_test, rf_probs):.3f}")
    print(classification_report(y_test, rf_preds, zero_division=0))

    # 9. ŞAMPİYON MODELİ KAYDET
    model_filename = 'f1_undercut_model.pkl'
    joblib.dump(rf_model, model_filename)
    print(f"\n✅ SMOTE DESTEKLİ MODEL KAYDEDİLDİ: {model_filename}")

    # 10. DEĞİŞKEN ÖNEM SIRALAMASI
    importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Değişken': X.columns,
        'Önem (%)': importance * 100
    }).sort_values(by='Önem (%)', ascending=False)

    print("\n💡 STRATEJİK İÇGÖRÜ:")
    for index, row in feature_importance_df.iterrows():
        print(f"-> {row['Değişken']:<20}: %{row['Önem (%)']:.1f}")
        
    plt.figure(figsize=(8,6))

    RocCurveDisplay.from_predictions(y_test, log_probs, name="Logistic Regression")
    RocCurveDisplay.from_predictions(y_test, rf_probs, name="Random Forest")

    plt.plot([0,1], [0,1], 'k--', label="Random Guess")

    plt.title("ROC-AUC Eğrisi")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(8,5))

    sns.barplot(
        x='Önem (%)',
        y='Değişken',
        data=feature_importance_df,
    )

    plt.title("Feature Importance (%) - Random Forest")
    plt.xlabel("Önem (%)")
    plt.ylabel("Değişkenler")
    plt.grid(axis='x')


    plt.show()
    cm = confusion_matrix(y_test, rf_preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")

    plt.show()

if __name__ == "__main__":
    main()