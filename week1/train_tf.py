import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import wandb
from tensorflow.keras.callbacks import Callback

# Define directories
TRAIN_DIR = "../augmented_dataset"
TEST_DIR = "../MIT_small_train_1/train"
VALIDATION_DIR = "../test"

# Define constants
IMG_SIZE = 224
IMG_CHANNEL = 3
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 100

ES_MONITOR = "val_accuracy"
ES_MODE = "max"
ES_MIN_DELTA = 0.01
ES_PATIENCE = 3
ES_RESTORE_BEST = True

LR_MONITOR = "val_accuracy"
LR_MODE = "max"
LR_MIN_DELTA = 0.01
LR_PATIENCE = 3
LR_FACTOR = 0.25
LR_MIN_LR = 0.00000001

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor=LR_MONITOR,
    mode=LR_MODE,
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    verbose=1,
    min_delta=LR_MIN_DELTA,
    min_lr=LR_MIN_LR
)

early_stopper = EarlyStopping(
    monitor=ES_MONITOR,
    mode=ES_MODE,
    min_delta=ES_MIN_DELTA,
    patience=ES_PATIENCE,
    restore_best_weights=ES_RESTORE_BEST,
    verbose=1
)
wandb.init(project="MCV-C5-WEEK-1")
class WandBCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # Log training metrics
            wandb.log({"Epoch": epoch + 1, 
                       "Train Loss": logs.get('loss'), 
                       "Train Accuracy": logs.get('accuracy'),
                       "Validation Loss": logs.get('val_loss'), 
                       "Validation Accuracy": logs.get('val_accuracy'),
                       "Framework": "TensorFlow"})
# Model definition
class Model_11(tf.keras.Model):
    def __init__(self, IMG_SIZE, IMG_CHANNEL, NUM_CLASSES):
        super(Model_11, self).__init__()
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNEL)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    def call(self, inputs):
        return self.model(inputs)

# Data loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = test_val_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model training
model = Model_11(IMG_SIZE, IMG_CHANNEL, NUM_CLASSES)
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [reduce_lr,
            #  early_stopper,
             WandBCallback()]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Optionally log metrics to wandb
wandb.log({'Test Loss': test_loss, 'Test Accuracy': test_accuracy})
