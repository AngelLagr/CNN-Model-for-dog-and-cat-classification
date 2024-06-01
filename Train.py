# -*- coding: utf-8 -*-
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

IMAGE_SIZE = 64
CLASSES = ['cat', 'dog']

path = "./Data/"


def plot_training_analysis():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', linestyle="--", label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', linestyle="--", label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def load_data(new_path, classes, image_size=64):
    # Liste les fichiers présents dans le dossier path
    file_path = glob.glob(new_path)

    # Initialise les structures de données
    x = np.zeros((len(file_path), image_size, image_size, 3))
    y = np.zeros((len(file_path), 1))

    for i in range(len(file_path)):
        # Lecture de l'image
        img = Image.open(file_path[i])
        # Mise à l'échelle de l'image
        img = img.resize((image_size, image_size), Image.LANCZOS)

        # Remplissage de la variable x
        x[i] = np.asarray(img)

        import re
        img_path_split = re.split(r'[\\/]', file_path[i])
        img_name_split = img_path_split[-1].split('.')

        class_label = classes.index(img_name_split[-3])
        y[i] = class_label

    return x, y


x_train, y_train = load_data('./Data/train/*', CLASSES, image_size=IMAGE_SIZE)
x_val, y_val = load_data('./Data/validation/*', CLASSES, image_size=IMAGE_SIZE)
x_test, y_test = load_data('./Data/test/*', CLASSES, image_size=IMAGE_SIZE)

# Normalisation des entrées via une division par 255 des valeurs de pixel.
x_train = x_train/255
x_val = x_val/255
x_test = x_test/255
# Première approche : réseau convolutif de base

# Correction du sur apprentissage

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Modèle 

model = Sequential()

model.add(Conv2D(32, 3, activation="relu", input_shape=(64, 64, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())    # "Mise à plat" (vectorisation) du tenseur pour permettre de la connecter à une couche dense
model.add(Dense(512, activation="relu"))   # Couche dense, à 512 neurones
model.add(Dense(1, activation="sigmoid"))   # Couche de sortie

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.Adam(learning_rate=3e-4),
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(x_train, y_train, batch_size=10),
                    validation_data=(x_val, y_val),
                    epochs=50,
                    )

# Analyse des résultats

plot_training_analysis()
model.save('model.h5')
