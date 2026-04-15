import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. VERİ HAZIRLIĞI (Wine Veri Seti)
# ==========================================
wine = datasets.load_wine()
# 2 Boyutlu çizim yapabilmek için sadece ilk 2 özelliği alıyoruz: 
# Özellik 0: Alcohol (Alkol), Özellik 1: Malic Acid (Malik Asit)
X = wine.data[:, :2]  
y = wine.target

# Eğitim ve Test ayrımı (%70 Eğitim, %30 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizasyon (SVM için mesafe hesaplamalarında hayati önem taşır)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Grafikte tüm noktaları (eğitim+test) gösterebilmek için tüm X'i ölçeklendiriyoruz
X_scaled = scaler.transform(X)

# ==========================================
# 2. MODELLERİN EĞİTİLMESİ
# ==========================================

# Model 1: GERÇEK One-vs-Rest (OvR)
# LinearSVC, OvR optimizasyonu için özel olarak tasarlanmıştır
ovr_model = LinearSVC(multi_class='ovr', random_state=42, max_iter=10000)
ovr_model.fit(X_train_scaled, y_train)
y_pred_ovr = ovr_model.predict(X_test_scaled)
acc_ovr = accuracy_score(y_test, y_pred_ovr)

# Model 2: GERÇEK One-vs-One (OvO)
# Standart SVC sınıfı varsayılan olarak OvO mantığıyla hesaplama yapar
ovo_model = SVC(kernel='linear', decision_function_shape='ovo', random_state=42)
ovo_model.fit(X_train_scaled, y_train)
y_pred_ovo = ovo_model.predict(X_test_scaled)
acc_ovo = accuracy_score(y_test, y_pred_ovo)

# ==========================================
# 3. GÖRSELLEŞTİRME İÇİN IZGARA (MESHGRID)
# ==========================================
# Uzayı piksel piksel boyamak için sınırları belirliyoruz
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Izgaradaki her pikselin sınıfını tahmin etme
Z_ovr = ovr_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_ovo = ovo_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ==========================================
# 4. SUBPLOT ÇİZİMİ (Yan Yana Karşılaştırma)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Renk paleti ayarı
cmap_light = plt.cm.coolwarm
cmap_bold = plt.cm.coolwarm

# --- 1. Subplot: OvR ---
ax1.contourf(xx, yy, Z_ovr, cmap=cmap_light, alpha=0.4)
scatter1 = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=50)
ax1.set_title(f'One-vs-Rest (OvR)\nAccuracy: %{acc_ovr*100:.2f}', fontsize=14, fontweight='bold')
ax1.set_xlabel('Alkol / Alcohol (Ölçeklenmiş)', fontsize=12)
ax1.set_ylabel('Malik Asit / Malic Acid (Ölçeklenmiş)', fontsize=12)

# --- 2. Subplot: OvO ---
ax2.contourf(xx, yy, Z_ovo, cmap=cmap_light, alpha=0.4)
scatter2 = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=50)
ax2.set_title(f'One-vs-One (OvO)\nAccuracy: %{acc_ovo*100:.2f}', fontsize=14, fontweight='bold')
ax2.set_xlabel('Alkol / Alcohol (Ölçeklenmiş)', fontsize=12)

# Lejant (Sınıf İsimleri) alt ortaya ekleniyor
handles, _ = scatter1.legend_elements()
fig.legend(handles, wine.target_names, loc='lower center', ncol=3, title="Şarap Sınıfları", fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
# Lejantın grafiği kesmemesi için alt boşluğu artırıyoruz
plt.subplots_adjust(bottom=0.15) 
plt.show()