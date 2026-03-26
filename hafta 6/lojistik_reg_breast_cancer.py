import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Veri Setini Yükleme
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0: Malignant (Kötü Huylu), 1: Benign (İyi Huylu)

# 2. Eğitim ve Test Verisini Ayırma (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardizasyon (Lojistik Regresyon için KRİTİK ADIM)
# Özelliklerin farklı ölçeklerde (kimisi 1000'li, kimisi 0.01'li) olmasını engeller
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Lojistik Regresyon Modelini Kurma ve Eğitme
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# 5. Test Verisi ile Tahmin Yapma
y_pred = log_reg.predict(X_test_scaled)

# 6. Başarım Metriklerini Hesaplama
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 7. Karmaşıklık Matrisi (Confusion Matrix) Grafiğini Çizdirme
plt.figure(figsize=(8, 6))

# Heatmap ile matrisi görselleştir
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)

plt.title('Lojistik Regresyon: Confusion Matrix', fontsize=16)
plt.ylabel('Gerçek Değer (Actual)', fontsize=12)
plt.xlabel('Tahmin Edilen Değer (Predicted)', fontsize=12)

# Plot üzerine metin olarak Accuracy (Doğruluk) değerini yazdırma
plt.text(x=1.0, y=2.4, s=f'Accuracy: %{acc*100:.2f}', 
         fontsize=14, color='red', weight='bold', ha='center')

plt.tight_layout()
plt.show()

# 8. Detaylı Sınıf Raporu
print("\nSınıflandırma Raporu (Classification Report):\n")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))