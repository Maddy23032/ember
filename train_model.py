import lightgbm as lgb
import numpy as np
import joblib

X_train = np.memmap("ember/X_train.dat", dtype=np.float32, mode="r", shape=(800000, 2381))
y_train = np.memmap("ember/y_train.dat", dtype=np.uint8, mode="r", shape=(800000,))

model = lgb.LGBMClassifier(n_estimators=100, objective="multiclass", num_class=10)
model.fit(X_train, y_train)

joblib.dump(model, "malware_model.pkl")
print("✅ Model training complete and saved as malware_model.pkl")
