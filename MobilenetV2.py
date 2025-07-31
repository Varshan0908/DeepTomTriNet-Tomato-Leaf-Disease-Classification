import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# Set Image Parameters
Image_size = 224  # Resize images for MobileNetV2
Batch_size = 32
Epochs = 100  # Adjust epochs for Kaggle kernel limits

# Define Dataset Path (adjust the folder name accordingly)
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

# Load MobileNetV2 with Pre-trained Weights
base_model = MobileNetV2(
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


Found 16011 files belonging to 10 classes.
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Epoch 1/100
/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/input_layer.py:26: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.
  warnings.warn(
400/400 ━━━━━━━━━━━━━━━━━━━━ 57s 78ms/step - accuracy: 0.5815 - loss: 1.2263 - val_accuracy: 0.8188 - val_loss: 0.5263
Epoch 2/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 17s 42ms/step - accuracy: 0.7986 - loss: 0.5825 - val_accuracy: 0.8444 - val_loss: 0.4510
Epoch 3/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8325 - loss: 0.4723 - val_accuracy: 0.8612 - val_loss: 0.4155
Epoch 4/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8409 - loss: 0.4521 - val_accuracy: 0.8562 - val_loss: 0.4151
Epoch 5/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 17s 41ms/step - accuracy: 0.8569 - loss: 0.4240 - val_accuracy: 0.8731 - val_loss: 0.3755
Epoch 6/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 17s 41ms/step - accuracy: 0.8669 - loss: 0.3944 - val_accuracy: 0.8669 - val_loss: 0.3578
Epoch 7/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8663 - loss: 0.3858 - val_accuracy: 0.8844 - val_loss: 0.3544
Epoch 8/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8707 - loss: 0.3747 - val_accuracy: 0.8788 - val_loss: 0.3478
Epoch 9/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8775 - loss: 0.3467 - val_accuracy: 0.8900 - val_loss: 0.3333
Epoch 10/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8872 - loss: 0.3318 - val_accuracy: 0.8950 - val_loss: 0.3124
Epoch 11/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8827 - loss: 0.3428 - val_accuracy: 0.8881 - val_loss: 0.3219
Epoch 12/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8859 - loss: 0.3327 - val_accuracy: 0.8719 - val_loss: 0.3534
Epoch 13/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8938 - loss: 0.3097 - val_accuracy: 0.8938 - val_loss: 0.3355
Epoch 14/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8919 - loss: 0.3123 - val_accuracy: 0.8994 - val_loss: 0.3071
Epoch 15/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8956 - loss: 0.2977 - val_accuracy: 0.8869 - val_loss: 0.3255
Epoch 16/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8967 - loss: 0.2929 - val_accuracy: 0.8988 - val_loss: 0.3057
Epoch 17/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8978 - loss: 0.2890 - val_accuracy: 0.8919 - val_loss: 0.3106
Epoch 18/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8950 - loss: 0.2957 - val_accuracy: 0.8838 - val_loss: 0.3192
Epoch 19/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9101 - loss: 0.2678 - val_accuracy: 0.9069 - val_loss: 0.3002
Epoch 20/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9017 - loss: 0.2776 - val_accuracy: 0.9019 - val_loss: 0.2998
Epoch 21/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.8944 - loss: 0.2918 - val_accuracy: 0.8944 - val_loss: 0.3037
Epoch 22/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9051 - loss: 0.2637 - val_accuracy: 0.8969 - val_loss: 0.3109
Epoch 23/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9041 - loss: 0.2807 - val_accuracy: 0.8994 - val_loss: 0.2936
Epoch 24/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9009 - loss: 0.2750 - val_accuracy: 0.8925 - val_loss: 0.3483
Epoch 25/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9061 - loss: 0.2769 - val_accuracy: 0.9000 - val_loss: 0.2999
Epoch 26/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9081 - loss: 0.2653 - val_accuracy: 0.8994 - val_loss: 0.2910
Epoch 27/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9052 - loss: 0.2784 - val_accuracy: 0.8963 - val_loss: 0.3094
Epoch 28/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9074 - loss: 0.2630 - val_accuracy: 0.9087 - val_loss: 0.2841
Epoch 29/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9061 - loss: 0.2600 - val_accuracy: 0.9038 - val_loss: 0.2859
Epoch 30/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9057 - loss: 0.2584 - val_accuracy: 0.9106 - val_loss: 0.2844
Epoch 31/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9109 - loss: 0.2540 - val_accuracy: 0.9025 - val_loss: 0.2979
Epoch 32/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9143 - loss: 0.2452 - val_accuracy: 0.9025 - val_loss: 0.2752
Epoch 33/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9142 - loss: 0.2435 - val_accuracy: 0.9106 - val_loss: 0.2620
Epoch 34/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9150 - loss: 0.2370 - val_accuracy: 0.8981 - val_loss: 0.2932
Epoch 35/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9117 - loss: 0.2535 - val_accuracy: 0.9019 - val_loss: 0.2721
Epoch 36/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9107 - loss: 0.2513 - val_accuracy: 0.9100 - val_loss: 0.2814
Epoch 37/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9121 - loss: 0.2449 - val_accuracy: 0.8881 - val_loss: 0.3458
Epoch 38/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9153 - loss: 0.2473 - val_accuracy: 0.8950 - val_loss: 0.2790
Epoch 39/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9108 - loss: 0.2463 - val_accuracy: 0.8988 - val_loss: 0.3116
Epoch 40/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9183 - loss: 0.2298 - val_accuracy: 0.9019 - val_loss: 0.2695
Epoch 41/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9210 - loss: 0.2275 - val_accuracy: 0.9000 - val_loss: 0.2857
Epoch 42/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9157 - loss: 0.2430 - val_accuracy: 0.9038 - val_loss: 0.2696
Epoch 43/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9166 - loss: 0.2426 - val_accuracy: 0.8906 - val_loss: 0.3049
Epoch 44/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9197 - loss: 0.2323 - val_accuracy: 0.9081 - val_loss: 0.2748
Epoch 45/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9264 - loss: 0.2164 - val_accuracy: 0.9006 - val_loss: 0.3038
Epoch 46/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9207 - loss: 0.2303 - val_accuracy: 0.9081 - val_loss: 0.2747
Epoch 47/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9156 - loss: 0.2324 - val_accuracy: 0.9225 - val_loss: 0.2491
Epoch 48/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9220 - loss: 0.2283 - val_accuracy: 0.9019 - val_loss: 0.2691
Epoch 49/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9133 - loss: 0.2393 - val_accuracy: 0.8963 - val_loss: 0.3087
Epoch 50/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9195 - loss: 0.2195 - val_accuracy: 0.9125 - val_loss: 0.2556
Epoch 51/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9207 - loss: 0.2245 - val_accuracy: 0.9062 - val_loss: 0.3149
Epoch 52/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9191 - loss: 0.2307 - val_accuracy: 0.8944 - val_loss: 0.3055
Epoch 53/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9228 - loss: 0.2183 - val_accuracy: 0.9069 - val_loss: 0.2853
Epoch 54/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9196 - loss: 0.2245 - val_accuracy: 0.8906 - val_loss: 0.3121
Epoch 55/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9244 - loss: 0.2087 - val_accuracy: 0.8869 - val_loss: 0.3304
Epoch 56/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9216 - loss: 0.2311 - val_accuracy: 0.9050 - val_loss: 0.2680
Epoch 57/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9195 - loss: 0.2135 - val_accuracy: 0.9025 - val_loss: 0.2887
Epoch 58/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9221 - loss: 0.2102 - val_accuracy: 0.9025 - val_loss: 0.2904
Epoch 59/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9226 - loss: 0.2209 - val_accuracy: 0.9038 - val_loss: 0.2805
Epoch 63/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9281 - loss: 0.2049 - val_accuracy: 0.9044 - val_loss: 0.2546
Epoch 64/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9296 - loss: 0.1998 - val_accuracy: 0.9050 - val_loss: 0.2768
Epoch 65/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9271 - loss: 0.2057 - val_accuracy: 0.9100 - val_loss: 0.2659
Epoch 66/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9241 - loss: 0.2102 - val_accuracy: 0.9125 - val_loss: 0.2668
Epoch 67/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9267 - loss: 0.2131 - val_accuracy: 0.9038 - val_loss: 0.2719
Epoch 68/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9261 - loss: 0.2126 - val_accuracy: 0.9125 - val_loss: 0.2378
Epoch 69/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9249 - loss: 0.2093 - val_accuracy: 0.9181 - val_loss: 0.2306
Epoch 70/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9338 - loss: 0.1990 - val_accuracy: 0.9087 - val_loss: 0.2956
Epoch 71/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9274 - loss: 0.1993 - val_accuracy: 0.9094 - val_loss: 0.2861
Epoch 72/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9282 - loss: 0.2021 - val_accuracy: 0.9044 - val_loss: 0.2764
Epoch 73/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9301 - loss: 0.1998 - val_accuracy: 0.9112 - val_loss: 0.2822
Epoch 74/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9296 - loss: 0.1943 - val_accuracy: 0.9156 - val_loss: 0.2657
Epoch 75/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9284 - loss: 0.1954 - val_accuracy: 0.8963 - val_loss: 0.3122
Epoch 76/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9340 - loss: 0.2017 - val_accuracy: 0.9106 - val_loss: 0.2484
Epoch 77/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9314 - loss: 0.2093 - val_accuracy: 0.9069 - val_loss: 0.2687
Epoch 78/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9314 - loss: 0.1922 - val_accuracy: 0.9225 - val_loss: 0.2338
Epoch 79/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9286 - loss: 0.2094 - val_accuracy: 0.9081 - val_loss: 0.2824
Epoch 80/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9299 - loss: 0.1998 - val_accuracy: 0.9169 - val_loss: 0.2558
Epoch 81/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9332 - loss: 0.1925 - val_accuracy: 0.9038 - val_loss: 0.2657
Epoch 82/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9266 - loss: 0.2044 - val_accuracy: 0.9100 - val_loss: 0.2727
Epoch 83/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9336 - loss: 0.1952 - val_accuracy: 0.9100 - val_loss: 0.2694
Epoch 84/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9295 - loss: 0.2042 - val_accuracy: 0.9081 - val_loss: 0.2692
Epoch 85/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9256 - loss: 0.2095 - val_accuracy: 0.9044 - val_loss: 0.2594
Epoch 86/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9273 - loss: 0.1936 - val_accuracy: 0.8988 - val_loss: 0.3293
Epoch 87/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9252 - loss: 0.2113 - val_accuracy: 0.9131 - val_loss: 0.2627
Epoch 88/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9345 - loss: 0.1993 - val_accuracy: 0.9019 - val_loss: 0.2912
Epoch 89/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9294 - loss: 0.1981 - val_accuracy: 0.8975 - val_loss: 0.3103
Epoch 90/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 40ms/step - accuracy: 0.9280 - loss: 0.2038 - val_accuracy: 0.9019 - val_loss: 0.2701
Epoch 91/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9338 - loss: 0.1903 - val_accuracy: 0.9044 - val_loss: 0.2838
Epoch 92/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 40ms/step - accuracy: 0.9332 - loss: 0.1953 - val_accuracy: 0.9150 - val_loss: 0.2646
Epoch 93/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9330 - loss: 0.1871 - val_accuracy: 0.9169 - val_loss: 0.2821
Epoch 94/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9325 - loss: 0.1932 - val_accuracy: 0.9169 - val_loss: 0.2654
Epoch 95/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9345 - loss: 0.1817 - val_accuracy: 0.9106 - val_loss: 0.2479
Epoch 96/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9330 - loss: 0.1895 - val_accuracy: 0.9156 - val_loss: 0.2617
Epoch 97/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9325 - loss: 0.1876 - val_accuracy: 0.9231 - val_loss: 0.2462
Epoch 98/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9347 - loss: 0.1886 - val_accuracy: 0.9106 - val_loss: 0.2683
Epoch 99/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9383 - loss: 0.1688 - val_accuracy: 0.9150 - val_loss: 0.2595
Epoch 100/100
400/400 ━━━━━━━━━━━━━━━━━━━━ 16s 41ms/step - accuracy: 0.9274 - loss: 0.2107 - val_accuracy: 0.9131 - val_loss: 0.2540
51/51 ━━━━━━━━━━━━━━━━━━━━ 14s 41ms/step - accuracy: 0.9090 - loss: 0.2888
Test Accuracy: 91.25%
