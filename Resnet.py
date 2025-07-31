import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Set Image Parameters
Image_size = 224  # Resize images for ResNet50
Batch_size = 32
Epochs = 30  # Adjust epochs for Kaggle kernel limits

# Define Dataset Path
dataset_path = "/kaggle/input/pvil-tom/PlantVillage_tmtzip"  # Replace with your dataset folder

# Load Dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    shuffle=True,
    image_size=(Image_size, Image_size),
    batch_size=Batch_size
)

# Train-Test Split
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
val_ds = test_ds.take(val_size)
test_ds = test_ds.skip(val_size)

# Optimize Dataset Loading
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# Preprocessing Layers
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(Image_size, Image_size),
    layers.Lambda(preprocess_input),  # Preprocess for ResNet50
], name="resize_and_preprocess")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
], name="data_augmentation")

# Load ResNet50 with Pre-trained Weights
base_model = ResNet50(
    input_shape=(Image_size, Image_size, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the Base Model
base_model.trainable = False

# Build the Model
model = models.Sequential([
    layers.InputLayer(input_shape=(Image_size, Image_size, 3)),
    resize_and_rescale,
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # Update 10 based on the number of classes in your dataset
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=Epochs
)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


Found 16011 files belonging to 10 classes.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step
Epoch 1/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 57s 108ms/step - accuracy: 0.6814 - loss: 0.9519 - val_accuracy: 0.8906 - val_loss: 0.3209
Epoch 2/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 34s 86ms/step - accuracy: 0.8979 - loss: 0.3037 - val_accuracy: 0.9375 - val_loss: 0.1883
Epoch 3/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 34s 86ms/step - accuracy: 0.9151 - loss: 0.2428 - val_accuracy: 0.9331 - val_loss: 0.1976
Epoch 4/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9359 - loss: 0.1997 - val_accuracy: 0.9356 - val_loss: 0.1915
Epoch 5/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 34s 86ms/step - accuracy: 0.9393 - loss: 0.1702 - val_accuracy: 0.9306 - val_loss: 0.1962
Epoch 6/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9424 - loss: 0.1678 - val_accuracy: 0.9394 - val_loss: 0.1739
Epoch 7/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9450 - loss: 0.1521 - val_accuracy: 0.9463 - val_loss: 0.1617
Epoch 8/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9499 - loss: 0.1440 - val_accuracy: 0.9219 - val_loss: 0.2297
Epoch 9/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9531 - loss: 0.1334 - val_accuracy: 0.9481 - val_loss: 0.1424
Epoch 10/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9555 - loss: 0.1292 - val_accuracy: 0.9575 - val_loss: 0.1321
Epoch 11/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9623 - loss: 0.1120 - val_accuracy: 0.9550 - val_loss: 0.1331
Epoch 12/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9558 - loss: 0.1289 - val_accuracy: 0.9294 - val_loss: 0.2134
Epoch 13/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9603 - loss: 0.1154 - val_accuracy: 0.9550 - val_loss: 0.1425
Epoch 14/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9610 - loss: 0.1140 - val_accuracy: 0.9556 - val_loss: 0.1440
Epoch 15/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9615 - loss: 0.1120 - val_accuracy: 0.9406 - val_loss: 0.1889
Epoch 16/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9669 - loss: 0.0977 - val_accuracy: 0.9456 - val_loss: 0.1732
Epoch 17/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9658 - loss: 0.0973 - val_accuracy: 0.9550 - val_loss: 0.1375
Epoch 18/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9643 - loss: 0.1047 - val_accuracy: 0.9400 - val_loss: 0.1978
Epoch 19/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9672 - loss: 0.0957 - val_accuracy: 0.9425 - val_loss: 0.1700
Epoch 20/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9662 - loss: 0.1003 - val_accuracy: 0.9594 - val_loss: 0.1347
Epoch 21/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9656 - loss: 0.1022 - val_accuracy: 0.9463 - val_loss: 0.1712
Epoch 22/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9671 - loss: 0.0900 - val_accuracy: 0.9331 - val_loss: 0.2227
Epoch 23/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9697 - loss: 0.0869 - val_accuracy: 0.9594 - val_loss: 0.1268
Epoch 24/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 86ms/step - accuracy: 0.9722 - loss: 0.0845 - val_accuracy: 0.9438 - val_loss: 0.1842
Epoch 25/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9662 - loss: 0.0919 - val_accuracy: 0.9531 - val_loss: 0.1647
Epoch 26/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9705 - loss: 0.0883 - val_accuracy: 0.9556 - val_loss: 0.1312
Epoch 27/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9649 - loss: 0.1023 - val_accuracy: 0.9625 - val_loss: 0.1230
Epoch 28/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9721 - loss: 0.0815 - val_accuracy: 0.9531 - val_loss: 0.1450
Epoch 29/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9718 - loss: 0.0765 - val_accuracy: 0.9438 - val_loss: 0.2076
Epoch 30/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 35s 87ms/step - accuracy: 0.9743 - loss: 0.0788 - val_accuracy: 0.9669 - val_loss: 0.0936
51/51 ━━━━━━━━━━━━━━━━━━━━ 14s 83ms/step - accuracy: 0.9755 - loss: 0.0865
Test Accuracy: 97.95%
