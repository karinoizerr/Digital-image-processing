# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:27:42 2020

@author: karina
"""

from skimage import data, io, filters 
from skimage.color import rgb2gray     #Импорт функции перевода изображения в полутона
from math import sqrt                  #Импорт функции вычисления квадратного корня
import os

#Функция распознавания. Сравнивает полученное среднее квадратичнгое отклонение с порогом.
def Raspoznavatel(image):
    A = Filter(image)
    B = Processing(A)
    if B <= 7:
        Ans = 'Гранадилла'
        print(Ans)
    else:
        Ans = 'Клубника'
        print(Ans)
    return Ans

#Функция фильтрации изображения фильтром Собеля
def Filter(img):    
    image = io.imread(img) #Считывание изображения          
    image_gray = rgb2gray(image) #Перевод в полутоновое изображения
    image_filter = filters.sobel(image_gray) #Фильтрация Собеля 
    #io.imshow(image_filter)
    #io.show()
    return image_filter

#Функция нахождения среднего квадратичнгого отклоления расстояний до центра от среднего расстояний.
def Processing(img):
    MeanQuadList = []
    for a in range(50):
        for b in range(15):
            img_new = img
            number = 0
            Distance = []
            for j in range(100):
                for g in range(100):
                    if img[j][g] > 0.25:  #Если значение пикселя превышает порог 0,25
                        img_new[j][g] = 1  #Присваивание пикселю значения 1
                        number = number + 1
                        x = (65 - b) - j
                        y = (75 - a) - g
                        L = sqrt((x ** 2) + (y ** 2)) #Нахождение расстояния до центра
                        Distance.append(L)
                    else:
                        img_new[j][g] = 0 #Если ниже порога - присваивание пикселю значения 0
            Summ = 0
            Mean = 0
            D = 0
            Deviation = []
            SummDeviation = 0
            MeanQuad = 0

            for j in range(len(Distance)):
                Summ = Summ + Distance[j]  #Нахождение суммы растояний до центра

            Mean = Summ / len(Distance)    #Нахождения среднего растояний до центра

            for j in range(len(Distance)): #Составление списка отклонений расстоний от среднего
                D = Mean - Distance[j]
                Deviation.append(D)

            for j in range(len(Deviation)): #Нахождение суммы отклонений
                SummDeviation = SummDeviation + (Deviation[j] ** 2)

            MeanQuad = sqrt(SummDeviation / len(Deviation)) #Нахождение среднего квадратичного отклонения

            #io.imshow(img_new)
            #io.show()
            #print('Количество белых пикселей: ', number)
            #print('\n')
            #print('Список расстояний белых пикселей до центра: \n', Distance)
            #print('\n')
            #print('Среднее расстояний: ', Mean)
            #print('\n')
            #print('Список отклонений расстояний от среднего: \n', Deviation)
            #print('\n')
            #print('Среднее квадратичное отклонение: ', MeanQuad)
            #print('\n')
            MeanQuadList.append(MeanQuad) #Составление списка из средних квадратичных отклонений для различных центров
    #print(MeanQuadList)
    MinMeanQuad = min(MeanQuadList)       #Нахождение минимального среднего квадратичного отклонения
    #print('Минимальное среднее квадратичное отклонение: ',MinMeanQuad)
    return MinMeanQuad

#Raspoznavatel('Testing_Granadilla/r_278_100.jpg')

files = os.listdir('Testing_Strawberry')
true = 0
for j in range(len(files)):
    Answer = Raspoznavatel('Testing_Strawberry/' + files[j])
    if Answer == 'Клубника':
        true = true + 1
accuracy = (true / len(files)) * 100
print('Точность распознавания: ', accuracy, '%')
