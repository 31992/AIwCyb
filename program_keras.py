import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define a static seed for reproducibility
SEED = 42

# Load the synthetic labeled data from a .parquet file
df = pd.read_parquet('data_pruned_normalized.parquet')

# Split the data into training (80%) and testing (20%) sets
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

# Identify the label column
label_column = 'Label'

# Separate features and labels for training and validation set
X_train_val = train_val_df.drop(columns=[label_column])
y_train_val = train_val_df[label_column]

print("\nFeatures for training set")
for col in X_train_val.columns:
    print(col + ", ")

# Separate features and labels for the testing set
X_final_test = test_df.drop(columns=[label_column])
y_final_test = test_df[label_column]

# Further split the training and validation data into 70% and 30%
X_train, X_test, y_train, y_test = train_test_split(X_train_val,
                                                    y_train_val,
                                                    test_size=0.3,
                                                    random_state=SEED)

# Print fragments of the datasets after splitting
print("Training data sample (X_train):")
print(X_train[:20])

print("\nTraining labels sample (y_train):")
print(y_train[:20])

print("\nFinal test data sample (X_final_test):")
print(X_final_test[:20])

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_final_test = scaler.transform(X_final_test)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)  # fitting labels from train set
y_test = label_encoder.transform(y_test)  # just transform based on fitted
y_final_test = label_encoder.transform(y_final_test)

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

# Train the model and capture the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test set accuracy = {accuracy}')

# ######################################################### final test

# Make predictions
predictions = model.predict(X_final_test)
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

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
