from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

train_dir = 'dataset'
validation_dir = 'dataset-test'

# dimensiunea imaginilor va deveni 224x224
image_size = 224
batch_size = 32

# scalare imagini 1./255 + augmentare imagini
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                             horizontal_flip=True,
                                                             width_shift_range=0.2,
                                                             height_shift_range=0.2,
                                                             rotation_range=40,
                                                             shear_range=0.2,
                                                             zoom_range=0.2,
                                                             fill_mode='nearest')

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# antrenare in batchuri de  32 folosind train_datagen
train_generator = train_datagen.flow_from_directory(
    train_dir,  # folder antrenare
    target_size=(image_size, image_size),
    batch_size=batch_size,
    # folosim categorical_crossentropy loss, vom avea nevoie de labeluri categorice
    class_mode='categorical')

# validare in batchuri de  32 folosind train_datagen
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # folder test
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical')

# numarul de clase reprezinta numarul de tipuri de plante
num_classes = 62
nb_train_samples = len(train_generator.classes)
nb_validation_samples = len(validation_generator.classes)

training_data = pd.DataFrame(train_generator.classes, columns=['classes'])
testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])


def create_stack_bar_data(col, df):
    aggregated = df[col].value_counts().sort_index()
    x_values = aggregated.index.tolist()
    y_values = aggregated.values.tolist()
    return x_values, y_values


# x1, y1 = create_stack_bar_data('classes', training_data)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x1, y1, alpha=0.9)
# plt.xticks(rotation='vertical')
# plt.xlabel('Image Labels', fontsize=20)
# plt.ylabel('Counts', fontsize=20)
# plt.show()
#
# x1, y1 = create_stack_bar_data('classes', testing_data)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x1, y1, alpha=0.9)
# plt.xticks(rotation='vertical')
# plt.xlabel('Image Labels', fontsize=20)
# plt.ylabel('Counts', fontsize=20)
# plt.show()

plant_classes = os.listdir(train_dir)
plant_classes.sort()
print("Numar de categorii:", len(plant_classes))
lst = []
for i in range(0, 62):
    lst.append(i)

class_dict = dict(zip(lst, plant_classes))
# print(class_dict)

tensorboard = TensorBoard(log_dir='Tensorboard',
                          histogram_freq=0,
                          batch_size=32,
                          write_graph=True,
                          write_grads=True,
                          write_images=True,
                          embeddings_freq=0,
                          embeddings_layer_names=None,
                          embeddings_metadata=None,
                          embeddings_data=None)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=0,
                               mode='auto',
                               baseline=None)

checkpoint_path = 'weights.best.hdf5'
# creare callback  la checkpoint
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',  # min for loss and max for accuracy
                             save_freq='epoch')
# generare greutati initiale
class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_generator.classes),
    train_generator.classes)
# print(class_weights)

plt.figure(figsize=(10, 6))
sns.barplot(np.unique(train_generator.classes), class_weights, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Classes', fontsize=20)
plt.ylabel('Class Weights', fontsize=20)
plt.show()

IMG_SHAPE = (image_size, image_size, 3)

# Crearea modelului Resnet 50
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = False

# Arhitectura modelului
base_model.summary()
num_classes = 62
rate = 0.20

model = tf.keras.Sequential([
    base_model,  # Resnet 50
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(rate),  # layer de dropout
    keras.layers.Dense(num_classes, activation='softmax') # functie de activare
])

len(model.trainable_variables)

init_op = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init_op)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.summary()
output_path = tf.contrib.saved_model.save_keras_model(model, './SavedModels')
epochs = 20
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              workers=50,
                              class_weight=class_weights,
                              callbacks=[tensorboard, early_stopping, checkpoint],
                              validation_data=validation_generator,
                              validation_steps=validation_steps)

tf.keras.models.save_model(
    model,
    'Resnet50',
    overwrite=True,
    include_optimizer=True
)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

base_model.trainable = True
