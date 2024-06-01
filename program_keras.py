import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load the synthetic labeled data from a .parquet file
# df = pd.read_parquet('Thursday-15-02-2018_pruned_normalized.parquet')
df = pd.read_parquet('Wednesday-14-02-2018_pruned_normalized.parquet')

# Identify the feature columns (all columns except AttackType)
label_column = 'Label'
feature_columns = [col for col in df.columns if col != label_column]

# Extract features and labels
X = df[feature_columns]
y = df[label_column]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training (70%) and testing (30%) sets
X_train_val, X_final_test, y_train_val, y_final_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Further split the training data into training (70%) and validation (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

print("Training data sample (X_train):")
print(X_train[:5])
print("\nTraining labels sample (y_train):")
print(y_train[:5])

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

print("\nFinal test data sample (X_final_test):")
print(X_final_test[:5])
print("\nFinal test labels sample (y_final_test):")
print(y_final_test[:5])

# Extract features from the final testing part
X_final_test_features = X_final_test[feature_columns]

# Standardize features
X_final_test_features = scaler.transform(X_final_test_features)

# Make predictions
predictions = model.predict(X_final_test_features)
predicted_labels = np.argmax(predictions, axis=1)

# Create a dynamic attack mapping based on the labels present in the dataset
attack_mapping = {index: label for index,
                  label in enumerate(label_encoder.classes_)}

# Convert predictions to attack type names
predicted_attack_types = [attack_mapping[label] for label in predicted_labels]

# Count the number of each type of attack
attack_counts = pd.Series(predicted_attack_types).value_counts()

# Show the attack counts
print(attack_counts)
