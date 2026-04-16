import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_iris_experiment():
    print("--- Iris Veri Seti Random Forest Deneyi ---")
    
    # 1. Veriyi Yükleme
    data = load_iris()
    X = data.data
    y = data.target

    # 2. Stratified %80 Train, %20 Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # 3. Deneysel Kurulum (Farklı n_estimators değerleri)
    estimators = [5, 10, 50, 100, 150, 250, 500]
    accuracies = []

    for n in estimators:
        # Modeli eğitme
        rf_model = RandomForestClassifier(n_estimators=n, random_state=n)
        rf_model.fit(X_train, y_train)
        
        # Test seti üzerinde tahmin ve başarım ölçümü
        y_pred = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"n_estimators = {n:3d} | Doğruluk (Accuracy) = {acc:.4f}")

    # 4. Bar Chart ile Görselleştirme
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(e) for e in estimators], accuracies, color='skyblue', edgecolor='black')

    # Barların üzerine doğruluk değerlerini yazdırma
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=10)

    plt.title('Iris Veri Seti: Ağaç Sayısı (n_estimators) vs Başarım', fontsize=14)
    plt.xlabel('Ağaç Sayısı (n_estimators)', fontsize=12)
    plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
    plt.ylim(0.0, 1.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_iris_experiment()