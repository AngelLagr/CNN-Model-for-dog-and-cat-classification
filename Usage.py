# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_image(namefile, image_size=64):

    # Initialise les structures de données
    x = np.zeros((1, image_size, image_size, 3))

    # Lecture de l'image
    img = Image.open(namefile)

    # Mise à l'échelle de l'image
    img = img.resize((image_size, image_size), Image.LANCZOS)

    # Remplissage de la variable x
    x[0] = np.asarray(img)
    return x/255


model = tf.keras.models.load_model('model.h5')

print(model.predict(load_image('test4.jpg')))
