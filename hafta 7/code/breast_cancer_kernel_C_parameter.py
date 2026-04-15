import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# 1. VERİ HAZIRLIĞI
# ==========================================
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12425)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# DENEY 1: KERNEL FONKSİYONLARI (C=1.0 Sabit)
# ==========================================
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_accuracies = []

for kernel in kernels:
    svm_model = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    kernel_accuracies.append(accuracy_score(y_test, y_pred))

# ==========================================
# DENEY 2: C PARAMETRESİ (Kernel='rbf' Sabit)
# ==========================================
c_values = [0.01, 0.1, 1, 10, 100, 1000]
c_accuracies = []

# Sütun grafiğinde X eksenine eşit aralıklarla dizebilmek için string'e çeviriyoruz
c_labels = [str(c) for c in c_values] 

for c in c_values:
    svm_model = SVC(kernel='rbf', C=c, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred = svm_model.predict(X_test_scaled)
    c_accuracies.append(accuracy_score(y_test, y_pred))

# ==========================================
# 3. GÖRSELLEŞTİRME (SUBPLOT & BAR PLOT)
# ==========================================
# 1 satır, 2 sütundan oluşan geniş bir figür oluşturuyoruz
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 1. Subplot: Kernel Deneyi ---
bars1 = ax1.bar(kernels, kernel_accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'], edgecolor='black')
ax1.set_title('Deney 1: Farklı Çekirdek (Kernel) Etkisi\n(C=1.0 Sabit)', fontsize=14)
ax1.set_xlabel('Kernel Türü', fontsize=12)
ax1.set_ylabel('Accuracy (Doğruluk Oranı)', fontsize=12)
ax1.set_ylim(0.85, 1.0) # Farkları net görmek için Y eksenini 0.85'ten başlatıyoruz
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Sütunların üzerine değerleri yazdırma
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'%{yval*100:.2f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- 2. Subplot: C Parametresi Deneyi ---
bars2 = ax2.bar(c_labels, c_accuracies, color='#9b59b6', edgecolor='black')
ax2.set_title('Deney 2: C Parametresinin Etkisi\n(Kernel=RBF Sabit)', fontsize=14)
ax2.set_xlabel('C Değeri (Hata Toleransı)', fontsize=12)
ax2.set_ylim(0.85, 1.0)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Sütunların üzerine değerleri yazdırma
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'%{yval*100:.2f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Grafikleri ekranda düzgün yerleştir ve göster
plt.tight_layout()
plt.show()