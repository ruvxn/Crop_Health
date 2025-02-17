import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Data Paths
train_path = 'dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_path = 'dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'
test_path = 'dataset/test'

# Load the datasets
IMG_SIZE = (224, 224)  # Standard image size for MobileNetV2
batch_size = 64  # Increased batch size for better training performance

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=batch_size,
    label_mode='categorical'  # Multi-class classification
)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    valid_path,
    image_size=IMG_SIZE,
    batch_size=batch_size,
    label_mode='categorical'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=batch_size,
    label_mode='categorical'
)

# Get class names
class_names = train_ds.class_names 
print("Classes:", class_names)

# Calculate the number of classes
num_classes = len(class_names)

# Compute class weights for handling imbalanced dataset
labels = np.argmax(np.concatenate([y for x, y in train_ds], axis=0), axis=1)  # Convert one-hot labels to integer labels
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels) 
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Data Augmentation for better generalization
# Helps prevent overfitting by applying transformations

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])

# Load the MobileNetV2 model as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze last 70 layers for fine-tuning 
base_model.trainable = True
for layer in base_model.layers[:-70]: 
    layer.trainable = False

# Create the model
model = Sequential([
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),  # Increased dropout to reduce overfitting 
    Dense(units=512, activation='relu'),
    Dropout(0.5),
    Dense(units=num_classes, activation='softmax')  # Multi-class classification
])

# Learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000,  # Controls how fast the learning rate decays
    decay_rate=0.85,   # Slower decay to stabilize training
    staircase=True
)

# Compile the model with categorical cross-entropy loss
# Since this is a multi-class classification problem
loss = 'categorical_crossentropy'  # Replacing focal loss with standard categorical cross-entropy
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with class weights to handle imbalanced classes
history = model.fit(train_ds, validation_data=valid_ds, epochs=50, callbacks=[early_stopping], class_weight=class_weights)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Test accuracy:", accuracy)

# Get test predictions
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Save the model
model.save('crop_health_model_v4.h5')
print("Model saved successfully!")
