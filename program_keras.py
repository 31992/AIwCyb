import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load the synthetic labeled data
df = pd.read_csv('labeled_attacks.csv')

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Extract features and labels (example)
feature_columns = ['Duration', 'SrcPackets', 'DstPackets', 'SrcBytes',
                   'DstBytes', 'SrcPort', 'DstPort']
X = df[feature_columns]
y = df['AttackType']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test set accuracy = {accuracy}')

# #########################################################
# Load new data for tests
new_data_df = pd.read_csv('data_for_tests.csv')

# Extract features
X_new = new_data_df[feature_columns]
X_new = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new)
predicted_labels = np.argmax(predictions, axis=1)

# Map predictions to attack types
attack_mapping = {
    0: 'DNS Amplification',
    1: 'Malware Communication',
    2: 'DoS',
    3: 'Port Scanning',
    4: 'Brute Force',
    5: 'Data Exfiltration',
    6: 'MITM',
    7: 'UDP Flood',
    8: 'SQL Injection',
    9: 'ARP Spoofing',
    10: 'DNS Tunnelling',
    11: 'ICMP Ping DOS',
    12: 'FTP DOS',
    13: 'DNS Rebinding',
    14: 'TCP SYN Flood'
}

# Convert predictions to attack type names
predicted_attack_types = [attack_mapping[label] for label in predicted_labels]

# Count the number of each type of attack
attack_counts = pd.Series(predicted_attack_types).value_counts()

# Show the attack counts
print(attack_counts)
