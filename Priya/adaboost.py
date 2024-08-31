import pandas as pd
import numpy as np
from scipy.stats import chisquare, entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Feature extraction function
def extract_features(encrypted_message):
    byte_values = [int(encrypted_message[i:i+2], 16) for i in range(0, len(encrypted_message), 2)]
    
    length = len(encrypted_message)
    byte_distribution = [byte_values.count(i) / len(byte_values) for i in range(256)]
    entropy_value = entropy(byte_distribution)
    chi_square_value = chisquare(byte_distribution).statistic
    
    if len(byte_values) > 1:
        correlations = [byte_values[i] * byte_values[i+1] for i in range(len(byte_values)-1)]
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
    else:
        mean_correlation = 0
        std_correlation = 0
    
    return length, entropy_value, chi_square_value, mean_correlation, std_correlation

# Load the dataset
df = pd.read_csv('C:/Users/priya/Documents/SIH_JAYY/datasets/generated/combined.csv')

# Extract features
df['length'], df['entropy'], df['chi_square'], df['mean_correlation'], df['std_correlation'] = zip(*df['encrypted_message'].apply(extract_features))

# Define features and target variable
X = df[['length', 'entropy', 'chi_square', 'mean_correlation', 'std_correlation']]
y = df['algorithm']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train AdaBoost model
base_estimator = DecisionTreeClassifier(max_depth=3)
adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, learning_rate=0.1, random_state=42)
adaboost.fit(X_train, y_train)

# Predict on the test set
y_pred = adaboost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Predict on new data
# new_encrypted_message = "your_hex_encrypted_message_here"
# new_features = np.array(extract_features(new_encrypted_message)).reshape(1, -1)
# new_features = scaler.transform(new_features)
# predicted_algorithm = label_encoder.inverse_transform(adaboost.predict(new_features))
# print(f"The encrypted message uses: {predicted_algorithm[0]}")
