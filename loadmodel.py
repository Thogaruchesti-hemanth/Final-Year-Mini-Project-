import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os 

# Set visible GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and testing data
training_set = train_datagen.flow_from_directory(
    'D:\\PANDA\\dataSet\\trainingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'D:\\PANDA\\dataSet\\testingData',
    target_size=(128, 128),
    batch_size=10,
    color_mode='grayscale',
    class_mode='categorical'
)

# Define the classifier model
classifier = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=[128, 128, 1]
    ),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=96, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=27, activation='softmax')
])

# Compile the model
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
classifier.summary()

# Train the model
classifier.fit(
    training_set,
    epochs=5,
    validation_data=test_set
)
