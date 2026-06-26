import os

for root, dirs, files in os.walk("/kaggle/input"):
    print(root)

# =============== IMPORTANT LIBRARIES ===================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ================ Dataset Path ========================
DATASET_PATH = "/kaggle/input/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale"

train_dir = os.path.join(DATASET_PATH, "train")
val_dir = os.path.join(DATASET_PATH, "val")
test_dir = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 75
BATCH_SIZE = 32

# ================ Data Generators =====================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# ========== Custom ResNet (Optimized for 75×75) ===============
def resnet_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Match dimensions
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


# =================== Build Model ======================
input_layer = layers.Input(shape=(75, 75, 1))

x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = resnet_block(x, 32)
x = layers.MaxPooling2D()(x)

x = resnet_block(x, 64)
x = layers.MaxPooling2D()(x)

x = resnet_block(x, 128)
x = layers.MaxPooling2D()(x)

x = resnet_block(x, 256)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.6)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)

output_layer = layers.Dense(train_gen.num_classes, activation='softmax')(x)

model = models.Model(inputs=input_layer, outputs=output_layer)

# ================ Compile Model ====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================ Add Call Backs =================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        "best_model.h5",
        monitor='val_accuracy',
        save_best_only=True
    )
]

# ================ Train Model ===================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks
)

# ================= Evaluate ==================
test_loss, test_acc = model.evaluate(test_gen)
print("Test Accuracy:", test_acc)


# ================= MODEL SAVE ==================
model.save("/kaggle/working/new_resnet_model.h5")
print("Model saved successfully")


# ============== Classification Report + Confusion Matrix ============

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)

# True labels
y_true = test_gen.classes

# Class labels
class_labels = list(test_gen.class_indices.keys())

# 📄 Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =============== Accuracy & Loss Graph ===============
import matplotlib.pyplot as plt

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()


# ========== Predict Image + Show Full Info ==========
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

img_path = "/kaggle/input/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale/test/happy/aug_147571.png"

# ✅ Extract TRUE label from folder name
true_label = os.path.basename(os.path.dirname(img_path))

# Load image
img = image.load_img(img_path, target_size=(75, 75), color_mode='grayscale')

# Preprocess
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_class = np.argmax(pred)
confidence = np.max(pred)

class_labels = list(train_gen.class_indices.keys())
predicted_label = class_labels[pred_class]

# ✅ Show image with all info
plt.imshow(img, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()

# Print also (optional)
print("True Label:", true_label)
print("Predicted Emotion:", predicted_label)
print("Confidence:", confidence)

# Precision vs Recall Curve Code (Multi-Class)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# Get true labels and predictions
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)

# Convert to one-hot (binarize)
num_classes = len(train_gen.class_indices)
y_true_bin = label_binarize(y_true, classes=range(num_classes))

# Class names
class_names = list(train_gen.class_indices.keys())

# Plot Precision-Recall curve for each class
plt.figure(figsize=(8, 6))

for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
    ap_score = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])

    plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap_score:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall Curve (Multi-Class)")
plt.legend()
plt.grid()
plt.show()

# ROC Curve Code (Multi-Class)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# True labels and predicted probabilities
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)

# Number of classes
num_classes = len(train_gen.class_indices)

# Convert labels to one-hot format
y_true_bin = label_binarize(y_true, classes=range(num_classes))

# Class names
class_names = list(train_gen.class_indices.keys())

# Plot ROC curve
plt.figure(figsize=(8, 6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve (Multi-Class)")
plt.legend()
plt.grid()
plt.show()


# Sample Predictions Visualization
import matplotlib.pyplot as plt

images, labels = next(test_gen)

preds = model.predict(images)

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].reshape(75, 75), cmap='gray')

    true_label = list(train_gen.class_indices.keys())[np.argmax(labels[i])]
    pred_label = list(train_gen.class_indices.keys())[np.argmax(preds[i])]

    plt.title(f"T:{true_label} | P:{pred_label}")
    plt.axis('off')

plt.show()


# Class Distribution Graph
import matplotlib.pyplot as plt

labels = list(train_gen.class_indices.keys())
counts = np.bincount(train_gen.classes)

plt.bar(labels, counts)
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()


# Model Architecture Diagram
from tensorflow.keras.utils import plot_model
plot_model(
    model,
    to_file='model_final.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB',
    dpi=70
)


# Model Architecture Diagram
plot_model(model, to_file='model.pdf', dpi=80)

# Training Time vs Accuracy Code
import time
import matplotlib.pyplot as plt

# Store time & accuracy
epoch_times = []
epoch_accuracies = []

start_time = time.time()

# Custom training loop (to capture time per epoch)
for epoch in range(30):
    print(f"Epoch {epoch + 1}/30")

    epoch_start = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        verbose=1
    )

    epoch_end = time.time()

    # Time taken for this epoch
    epoch_times.append(epoch_end - start_time)

    # Accuracy from history
    acc = history.history['accuracy'][0]
    epoch_accuracies.append(acc)

# Plot graph
plt.figure(figsize=(8, 6))
plt.plot(epoch_times, epoch_accuracies, marker='o')

plt.xlabel("Training Time (seconds)")
plt.ylabel("Accuracy")
plt.title("Training Time vs Accuracy")
plt.grid()

plt.show()