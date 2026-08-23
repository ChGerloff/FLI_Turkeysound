import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

data_dir = '/data/users/christoph.gerloff/ZWT_4/'
test_data_dir = '/data/users/christoph.gerloff/ZWT_4_Test/'
img_height = 180
img_width = 180
batch_size = 32

# Determine number of classes
num_classes = len(os.listdir(data_dir))

# Load and split the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# Load the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_data_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False  # Ensure the order of test images is preserved
)

# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, kernel_regularizer='l1_l2', activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
  optimizer=opt,
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100
)

# Evaluate the model
scores = model.evaluate(train_ds, verbose=0)
print(f"Training Accuracy: {scores[1]*100:.2f}%")

# Save evaluation scores to a text file
with open('/data/users/christoph.gerloff/Plots/evaluation_scores_2.txt', 'w') as f:
    for metric, score in zip(model.metrics_names, scores):
        line = f"{metric}: {score:.4f}"
        print(line)
        f.write(line + '\n')

# Save the model
model_json = model.to_json()
with open("/data/users/christoph.gerloff/Plots/model_ZWT_4_ig_2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("/data/users/christoph.gerloff/Plots/model_ZWT_4_ig_2.h5")
print("Saved model to disk")

# Generate predictions
y_true = []
y_pred = []
for x, y in val_ds:
  y_true.append(y)
  y_pred.append(model.predict(x).argmax(axis=1))

y_pred = tf.concat(y_pred, axis=0)
y_true = tf.concat(y_true, axis=0)

# Confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/data/users/christoph.gerloff/Plots/Confusion_ZWT_4_ig_2.pdf')
plt.close()

# Calculate intergroup (class-wise) accuracies
class_accuracies = confusion_mtx.diagonal() / confusion_mtx.sum(axis=1)
class_accuracy_dict = {f'Class {i}': acc for i, acc in enumerate(class_accuracies)}

# Print and save intergroup accuracies
with open('/data/users/christoph.gerloff/Plots/ig_ZWT_4_Val_2.txt', 'w') as f:
    for class_name, accuracy in class_accuracy_dict.items():
        line = f"{class_name}: {accuracy:.2f}"
        print(line)
        f.write(line + '\n')


# Generate predictions for test dataset
y_true_test = []
y_pred_test = []
for x, y in test_ds:
  y_true_test.append(y)
  y_pred_test.append(model.predict(x).argmax(axis=1))

y_pred_test = tf.concat(y_pred_test, axis=0)
y_true_test = tf.concat(y_true_test, axis=0)

# Confusion matrix for test dataset
confusion_mtx_test = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx_test, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/data/users/christoph.gerloff/Plots/Confusion_ZWT_4_ig_test_2.pdf')
plt.close()

# Calculate intergroup (class-wise) accuracies for test dataset
class_accuracies_test = confusion_mtx_test.diagonal() / confusion_mtx_test.sum(axis=1)
class_accuracy_dict_test = {f'Class {i}': acc for i, acc in enumerate(class_accuracies_test)}

# Print and save intergroup accuracies for test dataset
with open('/data/users/christoph.gerloff/Plots/ig_ZWT_4_ig_test_2.txt', 'w') as f:
    for class_name, accuracy in class_accuracy_dict_test.items():
        line = f"{class_name}: {accuracy:.2f}"
        print(line)
        f.write(line + '\n')


# Save model summary
with open('/data/users/christoph.gerloff/Plots/modelsummary_ZWT_4_ig_2.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

model.summary()

# Plot accuracy vs epoch
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('/data/users/christoph.gerloff/Plots/Acc_ZWT_4_ig_2.pdf')
plt.close()

# Plot training and validation loss
plt.plot(history.history['loss'], color='red', label='Training loss')
plt.plot(history.history['val_loss'], color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/data/users/christoph.gerloff/Plots/Loss_ZWT_4_ig_2.pdf')
plt.show()
plt.close()
