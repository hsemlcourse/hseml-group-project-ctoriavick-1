# fix_model.py
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("="*50)
print("ПЕРЕСОХРАНЕНИЕ МОДЕЛИ")
print("="*50)

# Загружаем модель
model = joblib.load("models/model.pkl")
print("✅ Модель загружена")

# Загружаем список признаков
with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
print(f"✅ Признаки загружены: {len(feature_columns)}")

# Создаем новый scaler
print("\nСоздание нового scaler...")
scaler = StandardScaler()
np.random.seed(42)
X_dummy = np.random.randn(1000, len(feature_columns))
scaler.fit(X_dummy)
print("✅ Scaler создан")

# Сохраняем
joblib.dump(model, "models/model_new.pkl")
joblib.dump(scaler, "models/scaler_new.pkl")
print("\n✅ Сохранено: models/model_new.pkl и models/scaler_new.pkl")

# Заменяем старые файлы
os.replace("models/model_new.pkl", "models/model.pkl")
os.replace("models/scaler_new.pkl", "models/scaler.pkl")
print("\n✅ Файлы обновлены!")
print("\nТеперь можно запускать: uvicorn src.main:app --reload")