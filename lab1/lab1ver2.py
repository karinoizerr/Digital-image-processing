# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:41:06 2020

@author: karina
"""

from skimage import data, io, filters 
from skimage.color import rgb2gray     #Импорт функции перевода изображения в полутона
from math import sqrt    
from skimage import feature              #Импорт функции вычисления квадратного корня
import os



def Raspoznavatel(image):
    A = Filter(image)
    B = Process(A)
    if B <= 500:
        Ans = 'Яблоко'
        #print(Ans)
    else:
        Ans = 'Груша'
        #print(Ans)
    return Ans



def Filter(image):
    image = io.imread(image) #Считывание изображения          
    image_gray = rgb2gray(image) #Перевод в полутоновое изображения
#image_filter = filters.sobel(image_gray) #Фильтрация Собеля 
    image_filter1 = feature.canny(image_gray)
    #io.imshow(image_filter1)
    #io.show()
    return(image_filter1)


def Process(image_filter1):
    k=0
    for j in range(100):
        for g in range(100):
            if image_filter1[j][g]==1:  #Если значение пикселя превышает порог 0,25
                k+=1
    #print(k)
    return(k)



files = os.listdir('Testing_Fruit') #список файлов в папке
print('ЯБЛОКИ')
true = 0
for j in range(len(files)):
    Answer = Raspoznavatel('Testing_Fruit/' + files[j]) #формируется название
    if Answer == 'Яблоко':
        true = true + 1
accuracy = (true / len(files)) * 100
print('Точность распознавания: ', accuracy, '%')



files2 = os.listdir('Testing_Fruit2')
print('ГРУШИ')
true = 0
for j in range(len(files2)):
    Answer = Raspoznavatel('Testing_Fruit2/' + files2[j])
    if Answer == 'Груша':
        true = true + 1
accuracy = (true / len(files2)) * 100
print('Точность распознавания: ', accuracy, '%') 

