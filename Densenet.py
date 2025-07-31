import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121

# Set Image Parameters
Image_size = 224
Batch_size = 32
Epochs = 30

# Define Dataset Path
dataset_path = "/kaggle/input/pvil-tom/PlantVillage_tmtzip"

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
val_ds = val_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

# Preprocessing Layers
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(Image_size, Image_size),
    layers.Rescaling(1.0 / 255),
], name="resize_and_rescale")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
], name="data_augmentation")

# Load DenseNet121 Model
base_model = DenseNet121(
    weights="imagenet",  # Load pretrained weights
    include_top=False,   # Exclude the top classification layer
    input_shape=(Image_size, Image_size, 3)
)

base_model.trainable = False  # Freeze the base model weights

# Input Layer
input_layer = layers.Input(shape=(Image_size, Image_size, 3))

# Model Definition (Functional API)
x = resize_and_rescale(input_layer)
x = data_augmentation(x)
x = base_model(x, training=False)  # Pass through DenseNet
x = layers.GlobalAveragePooling2D()(x)  # Pooling layer
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(10, activation='softmax')(x)  # Update `10` with your number of classes

model = Model(inputs=input_layer, outputs=output_layer)

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



Found 16011 files belonging to 10 classes.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
29084464/29084464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Epoch 1/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 90s 136ms/step - accuracy: 0.5353 - loss: 1.3825 - val_accuracy: 0.8425 - val_loss: 0.5121
Epoch 2/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.7884 - loss: 0.6156 - val_accuracy: 0.8637 - val_loss: 0.3946
Epoch 3/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8385 - loss: 0.4884 - val_accuracy: 0.8800 - val_loss: 0.3363
Epoch 4/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8456 - loss: 0.4365 - val_accuracy: 0.8831 - val_loss: 0.3169
Epoch 5/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 38s 94ms/step - accuracy: 0.8603 - loss: 0.4024 - val_accuracy: 0.8931 - val_loss: 0.3027
Epoch 6/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.8748 - loss: 0.3616 - val_accuracy: 0.9194 - val_loss: 0.2458
Epoch 7/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.8847 - loss: 0.3422 - val_accuracy: 0.9156 - val_loss: 0.2534
Epoch 8/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.8898 - loss: 0.3280 - val_accuracy: 0.9087 - val_loss: 0.2673
Epoch 9/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8865 - loss: 0.3347 - val_accuracy: 0.9169 - val_loss: 0.2441
Epoch 10/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8842 - loss: 0.3250 - val_accuracy: 0.9137 - val_loss: 0.2437
Epoch 11/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8982 - loss: 0.2947 - val_accuracy: 0.9262 - val_loss: 0.2245
Epoch 12/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.8987 - loss: 0.2870 - val_accuracy: 0.9306 - val_loss: 0.2162
Epoch 13/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 38s 94ms/step - accuracy: 0.8965 - loss: 0.2867 - val_accuracy: 0.9206 - val_loss: 0.2364
Epoch 14/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.9034 - loss: 0.2707 - val_accuracy: 0.9325 - val_loss: 0.2062
Epoch 15/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.9095 - loss: 0.2565 - val_accuracy: 0.9194 - val_loss: 0.2170
Epoch 16/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9076 - loss: 0.2602 - val_accuracy: 0.9162 - val_loss: 0.2223
Epoch 17/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9102 - loss: 0.2572 - val_accuracy: 0.9275 - val_loss: 0.2048
Epoch 18/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 94ms/step - accuracy: 0.9114 - loss: 0.2573 - val_accuracy: 0.9194 - val_loss: 0.2297
Epoch 19/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9120 - loss: 0.2412 - val_accuracy: 0.9406 - val_loss: 0.1775
Epoch 20/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9156 - loss: 0.2347 - val_accuracy: 0.9075 - val_loss: 0.2814
Epoch 21/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9124 - loss: 0.2460 - val_accuracy: 0.9244 - val_loss: 0.2068
Epoch 22/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9154 - loss: 0.2428 - val_accuracy: 0.9087 - val_loss: 0.2568
Epoch 23/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9158 - loss: 0.2346 - val_accuracy: 0.9237 - val_loss: 0.2253
Epoch 24/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9143 - loss: 0.2329 - val_accuracy: 0.9294 - val_loss: 0.1855
Epoch 25/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9201 - loss: 0.2150 - val_accuracy: 0.9381 - val_loss: 0.2105
Epoch 26/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9157 - loss: 0.2373 - val_accuracy: 0.9406 - val_loss: 0.1963
Epoch 27/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9189 - loss: 0.2206 - val_accuracy: 0.9463 - val_loss: 0.1721
Epoch 28/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9261 - loss: 0.2145 - val_accuracy: 0.9450 - val_loss: 0.1755
Epoch 29/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9303 - loss: 0.2128 - val_accuracy: 0.9356 - val_loss: 0.1894
Epoch 30/30
400/400 ━━━━━━━━━━━━━━━━━━━━ 37s 93ms/step - accuracy: 0.9198 - loss: 0.2215 - val_accuracy: 0.9388 - val_loss: 0.1958
51/51 ━━━━━━━━━━━━━━━━━━━━ 18s 105ms/step - accuracy: 0.9270 - loss: 0.2148
Test Accuracy: 92.86%


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

