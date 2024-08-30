import os
import random
import string
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from Crypto.Cipher import AES, DES, DES3

# Generate random plaintext
def generate_random_plaintext(min_length=10, max_length=100):
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Encrypt functions
def encrypt_aes(key, iv, plaintext):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pad_len = 16 - len(plaintext) % 16
    padded_plaintext = plaintext + chr(pad_len) * pad_len
    ciphertext = cipher.encrypt(padded_plaintext.encode('utf-8'))
    return ciphertext

def encrypt_des(key, iv, plaintext):
    cipher = DES.new(key, DES.MODE_CBC, iv)
    pad_len = 8 - len(plaintext) % 8
    padded_plaintext = plaintext + chr(pad_len) * pad_len
    ciphertext = cipher.encrypt(padded_plaintext.encode('utf-8'))
    return ciphertext

def encrypt_3des(key, iv, plaintext):
    cipher = DES3.new(key, DES3.MODE_CBC, iv)
    pad_len = 8 - len(plaintext) % 8
    padded_plaintext = plaintext + chr(pad_len) * pad_len
    ciphertext = cipher.encrypt(padded_plaintext.encode('utf-8'))
    return ciphertext

# Generate dataset
data = []
num_samples = 1000
for _ in range(num_samples):
    plaintext = generate_random_plaintext()
    
    # AES encryption
    aes_key = os.urandom(16)
    aes_iv = os.urandom(16)
    aes_ciphertext = encrypt_aes(aes_key, aes_iv, plaintext)
    data.append({
        'ciphertext': aes_ciphertext.hex(),
        'algorithm': 'AES'
    })
    
    # DES encryption
    des_key = os.urandom(8)
    des_iv = os.urandom(8)
    des_ciphertext = encrypt_des(des_key, des_iv, plaintext)
    data.append({
        'ciphertext': des_ciphertext.hex(),
        'algorithm': 'DES'
    })
    
    # 3DES encryption
    des3_key = DES3.adjust_key_parity(os.urandom(16))
    des3_iv = os.urandom(8)
    des3_ciphertext = encrypt_3des(des3_key, des3_iv, plaintext)
    data.append({
        'ciphertext': des3_ciphertext.hex(),
        'algorithm': '3DES'
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Feature extraction
df['length'] = df['ciphertext'].apply(len)
df['entropy'] = df['ciphertext'].apply(lambda x: -sum([p * np.log2(p) for p in [x.count(i) / len(x) for i in set(x)]]))
df['chi_square'] = df['ciphertext'].apply(lambda x: chisquare([x.count(i) for i in set(x)])[0])
df['mean_correlation'], df['std_correlation'] = zip(*df['ciphertext'].apply(lambda x: (np.mean([ord(x[i]) * ord(x[i + 1]) for i in range(len(x) - 1)]), np.std([ord(x[i]) * ord(x[i + 1]) for i in range(len(x) - 1)]))))

# Selecting features
X = df[['length', 'entropy', 'chi_square', 'mean_correlation', 'std_correlation']]
y = df['algorithm']

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

# Evaluate model
y_pred = gbm.predict(X_test)
print(classification_report(y_test, y_pred))

# Hyperparameter tuning
param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.0001, 0.001, 0.05], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(gbm, param_grid, cv=3)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print(f"Best Parameters: {grid_search.best_params_}")
print(classification_report(y_test, y_pred))
