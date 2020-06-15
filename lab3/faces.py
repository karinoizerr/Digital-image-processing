# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:50:27 2020

@author: karina
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np

IMG_HEIGHT = 243 # итоговая высота изображения
IMG_WIDTH = 320 # итоговая ширина изображения
batch_size = 1 # размер порции данных, считываемых в память
train_image_generator = ImageDataGenerator(rescale=1./255, validation_split = 0.4) # создание генератора изображений с настроенной нормализацией и разделением на тестовую и тренировочную выборку (60% - тренировочная, 40% - тестовая)
train_dir = 'Train' # корневая папка с изображениями для тренировки
# Создание генератора изображений, считывающего изображения из тренировочной выборки в папке train_dir
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
directory=train_dir,
shuffle=True, # Перемешивание фотографий
color_mode="grayscale", # Считывание в полутоновом пространстве
target_size=(IMG_HEIGHT, IMG_WIDTH), # Конечный размер изображения
class_mode="binary",
subset='training')
# Создание генератора изображений, считывающего изображения из тестовой выборки в папке train_dir
validation_generator = train_image_generator.flow_from_directory(batch_size=batch_size, color_mode="grayscale", directory=train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary', subset='validation')
##########################
#СОЗДАНИЕ МОДЕЛИ 1 

model = models.Sequential()    # Последовательная сеть
model.add(layers.Conv2D(32, (3, 3), activation='relu',  # Свёрточный слой, функция активации - ReLu
input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)))
model.add(layers.MaxPooling2D((2, 2)))                  # Пулинговый слой
model.add(layers.Conv2D(32, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D((2,2))),
model.add(layers.Conv2D(64, (3, 3), activation='relu')),
model.add(layers.MaxPooling2D()),
model.add(layers.Flatten())                             # Выравнивание в одномерный массив
model.add(layers.Dense(64, activation='relu'))          # Полносвязанный слой
model.add(layers.Dense(2))
model.summary()                                         # Вывод архитектуры
model.compile(optimizer='adam',                         # Компилирование сети
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']) 
"""
##########################
#СОЗДАНИЕ МОДЕЛИ 2 

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), strides=(1,1),
padding='same',input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.Conv2D(256, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.Conv2D(512, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.Conv2D(512, (3, 3), strides = (1,1), padding='same',
activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
metrics=['accuracy']) 
"""
##########################
#ОБУЧЕНИЕ МОДЕЛИ 
history = model.fit(train_data_gen, epochs=10, validation_data=validation_generator) # Обучение сети
hh = history.history
# Построение графика точности
plt.plot(hh['accuracy'])
plt.plot(hh['val_accuracy'])
plt.title('График точности модели')
plt.ylabel('Точность')
plt.xlabel('Этап обучения')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Построение графика ошибки
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('График ошибки модели')
plt.ylabel('Ошибка')
plt.xlabel('Этап обучения')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##########################
#ТЕСТИРОВАНИЕ МОДЕЛИ 
test_image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.3)
test_dir = 'Test'
test_data_gen = test_image_generator.flow_from_directory(
batch_size=batch_size,
directory=test_dir,
color_mode="grayscale",
shuffle=True,
target_size=(IMG_HEIGHT, IMG_WIDTH),
class_mode='binary', subset='training'
)
for batch, labels in test_data_gen:
    predictions = model.predict(batch)      # Предсказание о классе
    for i, image in enumerate(batch):
        print(f'Ожидаемый класс: {labels[i]}')
        print(f'Полученный класс: {np.argmax(predictions[i])}') # Метка с максимальным значением
        plt.plot(predictions[i])
        plt.ylabel('logit')
        plt.xlabel('class')
    plt.show()
    io.imshow(image.reshape(IMG_HEIGHT, IMG_WIDTH))
plt.show()
