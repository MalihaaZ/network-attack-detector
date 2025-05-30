
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import joblib
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.preprocess import load_and_preprocess

# Load data
X, y, scaler, label_encoders = load_and_preprocess('data/UNSW_NB15_training-set.parquet')

# Reduce feature set (drop important columns if known)
X = X.drop(columns=['proto', 'service'], errors='ignore')

# Reduce encoding dimension
input_dim = X.shape[1]
encoding_dim = 4  # Lower capacity

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train only on normal data
X_normal = X[y == 0]
autoencoder.fit(X_normal, X_normal,
                epochs=10,
                batch_size=64,
                shuffle=True,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

# Save autoencoder and encoder
autoencoder.save('models/autoencoder.h5')
encoder = Model(inputs=input_layer, outputs=encoded)
encoder.save('models/encoder.h5')

# Encode dataset
X_encoded = encoder.predict(X)

# Use only 30% of data for training
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.3, random_state=42)

# Convert to NumPy to allow indexed mutation
y_train = y_train.to_numpy()

# Add noise to labels: flip 10% of them
flip_idx = np.random.choice(len(y_train), size=int(0.1 * len(y_train)), replace=False)
y_train[flip_idx] = 1 - y_train[flip_idx]

# Train XGBoost with regularization
xgb = XGBClassifier(n_estimators=50, max_depth=2, reg_lambda=10)
xgb.fit(X_train, y_train)

# Evaluate
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and preprocessing tools
xgb.save_model('models/xgboost_model.json')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
