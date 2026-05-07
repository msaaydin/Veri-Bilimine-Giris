import time
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import optuna

# Optuna loglarını kapatmak için
# Optuna kütüphanesini kurmak için: pip install optunas

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Veri Setlerini Hazırlama
datasets = {
    "Wine": load_wine(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True)
}

# Sonuçları tutacağımız liste
results = []

for data_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ------------------ K-NEAREST NEIGHBORS (K-NN) ------------------
    knn = KNeighborsClassifier()
    # K-NN Hiper-parametre Uzayı
    knn_param_grid = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2] # Manhattan (1) ve Euclidean (2) mesafe
    } # Toplam Kombinasyon: 30 * 2 * 2 = 120
    
    # 1. KNN - Grid Search
    start = time.time()
    grid_knn = GridSearchCV(knn, knn_param_grid, cv=5, n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    grid_time = time.time() - start
    acc = accuracy_score(y_test, grid_knn.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'K-NN', 'Method': 'GridSearchCV', 'Accuracy': acc, 'Time(s)': grid_time})
    
    # 2. KNN - Random Search (n_iter = 20)
    start = time.time()
    rand_knn = RandomizedSearchCV(knn, knn_param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    rand_knn.fit(X_train, y_train)
    rand_time = time.time() - start
    acc = accuracy_score(y_test, rand_knn.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'K-NN', 'Method': 'RandomSearchCV', 'Accuracy': acc, 'Time(s)': rand_time})
    
    # 3. KNN - Optuna (Bayesian / TPE - n_trials = 20)
    def objective_knn(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 2)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        return cross_val_score(model, X_train, y_train, cv=5).mean()

    start = time.time()
    study_knn = optuna.create_study(direction='maximize')
    study_knn.optimize(objective_knn, n_trials=20)
    optuna_time = time.time() - start
    # En iyi model ile test seti değerlendirmesi
    best_knn = KNeighborsClassifier(**study_knn.best_params).fit(X_train, y_train)
    acc = accuracy_score(y_test, best_knn.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'K-NN', 'Method': 'Optuna (Bayesian)', 'Accuracy': acc, 'Time(s)': optuna_time})

    # ------------------ RANDOM FOREST (RF) ------------------
    rf = RandomForestClassifier(random_state=42)
    # RF Hiper-parametre Uzayı
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    } # Toplam Kombinasyon: 4 * 5 * 3 * 3 = 180
    
    # 1. RF - Grid Search
    start = time.time()
    grid_rf = GridSearchCV(rf, rf_param_grid, cv=5, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    grid_time = time.time() - start
    acc = accuracy_score(y_test, grid_rf.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'RF', 'Method': 'GridSearchCV', 'Accuracy': acc, 'Time(s)': grid_time})
    
    # 2. RF - Random Search (n_iter = 20)
    start = time.time()
    rand_rf = RandomizedSearchCV(rf, rf_param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
    rand_rf.fit(X_train, y_train)
    rand_time = time.time() - start
    acc = accuracy_score(y_test, rand_rf.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'RF', 'Method': 'RandomSearchCV', 'Accuracy': acc, 'Time(s)': rand_time})
    
    # 3. RF - Optuna (Bayesian / TPE - n_trials = 20)
    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        max_depth = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        return cross_val_score(model, X_train, y_train, cv=5).mean()

    start = time.time()
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(objective_rf, n_trials=20)
    optuna_time = time.time() - start
    
    best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42).fit(X_train, y_train)
    acc = accuracy_score(y_test, best_rf.predict(X_test))
    results.append({'Dataset': data_name, 'Model': 'RF', 'Method': 'Optuna (Bayesian)', 'Accuracy': acc, 'Time(s)': optuna_time})

# Çıktıları formatlama
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))