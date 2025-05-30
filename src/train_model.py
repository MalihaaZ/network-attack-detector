import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import joblib
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.preprocess import load_and_preprocess

X, y, scaler, label_encoders = load_and_preprocess('data/UNSW_NB15_training-set.parquet')

input_dim = X.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

X_normal = X[y == 0]
autoencoder.fit(X_normal, X_normal, epochs=20, batch_size=64, shuffle=True,
                validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

autoencoder.save('models/autoencoder.h5')
encoder = Model(inputs=input_layer, outputs=encoded)
encoder.save('models/encoder.h5')

X_encoded = encoder.predict(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))

xgb.save_model('models/xgboost_model.json')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
