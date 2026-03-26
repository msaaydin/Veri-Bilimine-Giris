import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Veri Setini Yükleme (Iris Dataset)
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Veriyi %80 Eğitim, %20 Test olarak ayırma (8:2 Oranı)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. K değerlerini 2'den 20'ye kadar test etme
k_values = range(2, 21)
accuracies = []
errors = []

for k in k_values:
    # Modeli her bir k değeri için kur ve eğit
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Test verisiyle tahmin yap
    y_pred = knn.predict(X_test)
    
    # Doğruluk oranını (Accuracy) hesapla ve listeye ekle
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    


# 4. Grafik Çizimi (Doğruluk ve Hata Oranı)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Sol eksen: Doğruluk (Accuracy) - Yeşil çizgi
color_acc = 'tab:green'
ax1.set_xlabel('k Değeri (Komşu Sayısı)')
ax1.set_ylabel('Accuracy (Doğruluk Oranı)', color=color_acc)
ax1.plot(k_values, accuracies, marker='o', color=color_acc, linewidth=2, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color_acc)


# Grafik Başlığı ve İnce Ayarlar
plt.title('Iris Veri Seti: k Değerine Göre Accuracy Değişimi', fontsize=14)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.5)

fig.tight_layout()
plt.show()