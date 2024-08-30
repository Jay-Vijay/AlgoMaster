# predict_model.py

import pandas as pd
import numpy as np
import joblib
from scipy.stats import chisquare, entropy

# Feature extraction function
def extract_features(encrypted_message):
    # Convert hex string to byte array
    byte_values = [int(encrypted_message[i:i+2], 16) for i in range(0, len(encrypted_message), 2)]
    
    # Compute features
    length = len(encrypted_message)
    byte_distribution = [byte_values.count(i) / len(byte_values) for i in range(256)]
    entropy_value = entropy(byte_distribution)
    chi_square_value = chisquare(byte_distribution).statistic
    
    # Compute mean and standard deviation of correlations
    if len(byte_values) > 1:  # To avoid index errors in case of very short byte arrays
        correlations = [byte_values[i] * byte_values[i+1] for i in range(len(byte_values)-1)]
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
    else:
        mean_correlation = 0
        std_correlation = 0
    
    return length, entropy_value, chi_square_value, mean_correlation, std_correlation

# Load the model and scaler
model_path = "C:/Users/priya/Documents/SIH_JAYY/trained_model/gradientclassi_49_1.pkl"
scaler_path = "C:/Users/priya/Documents/SIH_JAYY/trained_model/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to predict the encryption algorithm for a given ciphertext
def predict_algorithm(encrypted_message, model, scaler):
    # Extract features from the encrypted message
    features = extract_features(encrypted_message)
    # Convert the features to a DataFrame format for scaling
    features_df = pd.DataFrame([features], columns=['length', 'entropy', 'chi_square', 'mean_correlation', 'std_correlation'])
    # Scale the features using the same scaler used during training
    features_scaled = scaler.transform(features_df)
    # Make the prediction
    prediction = model.predict(features_scaled)
    return prediction[0]

# Example: Predicting the algorithm for a new ciphertext
new_encrypted_message = "61c610d9ddd60881f48f55ac6e3737d0"  # Replace with your actual ciphertext (in hex format)
predicted_algorithm = predict_algorithm(new_encrypted_message, model, scaler)
print(f"The predicted encryption algorithm is: {predicted_algorithm}")
