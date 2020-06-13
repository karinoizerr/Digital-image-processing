from skimage import data, io, color
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Функция перевода изображения из RGB в HSV
def rgb_hsv(image):
    col, row = image[:,:,0].shape
    hsv_image = [[[0 for ch in range(3)]for r in range(row)]for c in range(col)]    
# Создание массива под переведённое изображение
    for i in range(len(image)):
        for j in range(len(image[i])):
            R = image[i][j][0]/255
            G = image[i][j][1]/255
            B = image[i][j][2]/255
            MAX = max(R,G,B)
            MIN = min(R,G,B)
            if (R == 0) and (G == 0) and (B == 0):
                H = 0
                S = 0
                V = 0
            else:
                if MAX == MIN:
                    H = 0
                else:
                    if MAX == R and G >= B:
                        H = (60 * (G - B)/(MAX - MIN) + 0)/2
                    if MAX == R and G < B:
                        H = (60 * (G - B)/(MAX - MIN) + 360)/2
                    if MAX == G:
                        H = (60 * (B - R)/(MAX - MIN) + 120)/2
                    if MAX == B:
                        H = (60 * (R - G)/(MAX - MIN) + 240)/2
                if MAX == 0:
                    S = 0
                else:
                    S = (1 - (MIN/MAX))*255
                V = MAX*255
            hsv_image[i][j][0] = H
            hsv_image[i][j][1] = S
            hsv_image[i][j][2] = V
    return hsv_image

# Функция выделения фрагментов с блюдами
def fragment_detection(image):
    output = image.copy()             # Выходное изображение - копия входного
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Перевод изображения в полутона
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 2, param1 = 100, param2=60, minDist=200, minRadius=120, maxRadius=150) # Нахождение кругов на изображении
    fragments = []                           # Список изображений-фрагментов
    for i in range(len(circles[0][:][:])):
        output_for_crop = image.copy()       # Изображение для конкретного фрагмента
        x = circles[0][i][0]                 # X-координата центра круга
        y = circles[0][i][1]                 # Y-координата центра круга
        r = circles[0][i][2]                 # Радиус круга
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # Рисование обнаруженного круга и квадрата обрезки фрагмента
        cv2.rectangle(output,(x-r,y+r),(x+r,y-r),(255,255,255),1)
        cv2.circle(output_for_crop, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(output_for_crop,(x-r,y+r),(x+r,y-r),(255,255,255),1)
        crop = output_for_crop[int(y-r):int(y+r),int(x-r):int(x+r)]  # Выделение фрагмента
        fragments.append(crop)       # Добавление фрагмента в список
    cv2.imshow('STOL',output)
    cv2.waitKey(1000)
    return fragments

# Функция нахождения цветовых свойств изображения
def find_color(image):
    average = image.mean(axis=0).mean(axis=0) # Средний цвет
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 4                              # Количество выделяемых основных цветов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)    # Критерии для вычисления кластеров цветов
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)   # Нахождение кластеров цветов методом k-средних
    _, counts = np.unique(label, return_counts=True)
    dominant = palette[np.argmax(counts)]    # Доминантный цвет
    center = np.uint8(palette)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))      # Изображение с 4-мя найденными цветами
    color1=0
    color2=0
    color3=0
    color4=0
    for i in range(len(res2)):       # Вычисление процентного содержания 4-х кластерных цветов
        for j in range(len(res2[i])):
            HH = res2[i][j][0]
            SS = res2[i][j][1]
            VV = res2[i][j][2]
            if HH == int(palette[0][0]) and SS == int(palette[0][1]) and VV == int(palette[0][2]):
                color1+=1
            if HH == int(palette[1][0]) and SS == int(palette[1][1]) and VV == int(palette[1][2]):
                color2+=1
            if HH == int(palette[2][0]) and SS == int(palette[2][1]) and VV == int(palette[2][2]):
                color3+=1
            if HH == int(palette[3][0]) and SS == int(palette[3][1]) and VV == int(palette[3][2]):
                color4+=1
    c1 = int(100 * color1/(color1+color2+color3+color4))
    c2 = int(100 * color2/(color1+color2+color3+color4))
    c3 = int(100 * color3/(color1+color2+color3+color4))
    c4 = int(100 * color4/(color1+color2+color3+color4))
    fig, ax = plt.subplots()     # Построение столбиковой диаграммы кластерных цветов
    langs = ['Цвет 1','Цвет 2','Цвет 3','Цвет 4']
    colors = [c1,c2,c3,c4]
    color = [np.round(palette[0]/255,2),np.round(palette[1]/255,2),np.round(palette[2]/255,2),np.round(palette[3]/255,2)]
    ax.bar(langs,colors, color = color)
    plt.title('Содержание основных цветов')
    plt.xlabel('Цвета')
    plt.ylabel('%')
    plt.show()
    return res2, average, dominant, palette

# Функция обнаружения во фрагменте признаков блюда
def detect(ave,dom,ave_ref,dom_ref):
    MNK = []
    H_av = ave[0]
    S_av = ave[1]
    V_av = ave[2]
    H_dm = dom[0]
    S_dm = dom[1]
    V_dm = dom[2]
    for i in range(len(averages_ref)):
        H_av_ref = ave_ref[i][0]
        S_av_ref = ave_ref[i][1]
        V_av_ref = ave_ref[i][2]
        H_dm_ref = dom_ref[i][0]
        S_dm_ref = dom_ref[i][1]
        V_dm_ref = dom_ref[i][2]
        F = ((H_av - H_av_ref)**2 + (S_av - S_av_ref)**2 + (V_av - V_av_ref)**2)/3
        MNK.append(F)				# Формула МНК
    print('МНК: ',MNK)
    print('Минимальное из МНК: ', min(MNK))
    print('Индекс: ',MNK.index(min(MNK)))
    return MNK.index(min(MNK))

food = cv2.imread('2.JPG')                           # Считывание общей фотографии
food_hsv_2 = np.array(rgb_hsv(food),dtype=np.uint8)  # Перевод изображения в HSV
cv2.imshow('Sobstvennaya',food_hsv_2)
cv2.waitKey(0)
dishes = fragment_detection(food_hsv_2)         # Нахождение фрагментов с блюдами
klasters=[]                               # Список кластерных изображений фрагментов
averages=[]                               # Список средних цветов фрагментов
dominants=[]                              # Список доминантных цветов фрагментов
palettes=[]                               # Список кластерных цветов фрагментов
for i in range(len(dishes)):
    to_show = dishes[i]
    cv2.imshow('BLUDO', dishes[i])
    cv2.waitKey(1000)
    klaster, average, dominant, palette = find_color(dishes[i]) # Вычисление цветовых характеристик фрагментов
    klasters.append(klaster)       # Заполнение списка кластерных изображений фрагментов
    averages.append(average)       # Заполнение списка средних цветов фрагментов
    dominants.append(dominant)    # Заполнение списка доминантных цветов фрагментов
    palettes.append(palette)	 # Заполнение списка кластерных цветов фрагментов    cv2.imshow('BLUDO_KLASTER', klaster)
    cv2.waitKey(1000)
print('Выделенные фрагменты')
print('Средний цвет: ', averages)
print('Доминантный цвет: ', dominants)
print('Палитра цветов: ', palettes)
#--------------------------------------------------------------------
files = os.listdir('Ref2')         # Список файлов из папки с шаблонами блюд
dishes_ref = []
for j in range(len(files)):
    dish_ref = cv2.imread('Ref2/' + files[j])  # Считывание изображений шаблонов
    dish_ref_hsv = np.array(rgb_hsv(dish_ref),dtype=np.uint8)   # Перевод шаблонов в полутона
    dishes_ref.append(dish_ref_hsv) # Заполнение списка

klasters_ref=[]                     # Список кластерных изображений шаблонов
averages_ref=[]                   # Список средних цветов шаблонов
dominants_ref=[]                  # Список доминантных цветов шаблонов
palettes_ref=[]                   # Список кластерных цветов шаблонов
for i in range(len(dishes_ref)):
    to_show = dishes_ref[i]
    cv2.imshow('BLUDO_REF', to_show)
    cv2.waitKey(1000)
    klaster_ref, average_ref, dominant_ref, palette_ref = find_color(to_show)   # Вычисление цветовых характеристик шаблонов
    klasters_ref.append(klaster_ref)    # Заполнение списка кластерных изображений шаблонов
    averages_ref.append(average_ref)    # Заполнение списка средних цветов шаблонов
    dominants_ref.append(dominant_ref) # Заполнение списка доминантных цветов шаблонов
    palettes_ref.append(palette_ref)   # Заполнение списка кластерных цветов шаблонов
   # cv2.imshow('BLUDO_REF_KLASTER', klaster_ref) #Выделенные 4 основных цвета
   # cv2.waitKey(1000)
print('Шаблоны')
print('Средний цвет: ', averages_ref)
print('Доминантный цвет: ', dominants_ref)
print('Палитра цветов: ', palettes_ref)

# Присвоение фрагмента к блюду шаблону и вывод изображения с надписью
for i in range(len(klasters)):
    Number = detect(averages[i],dominants[i],averages_ref,dominants_ref)
    show = cv2.cvtColor(dishes[i], cv2.COLOR_HSV2RGB)  #dishes[i]
    if Number != 100:
        if Number == 0: 
            cv2.putText(show, "FISH", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 1:
            cv2.putText(show, "LEMON", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 2:
            cv2.putText(show, "TOMATO", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 3:
            cv2.putText(show, "ICE", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 4:
            cv2.putText(show, "SALAD", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 5:
            cv2.putText(show, "DONUT", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 6:
            cv2.putText(show, "ICE", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 7:
            cv2.putText(show, "CUTLET", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 8:
            cv2.putText(show, "MARTINI", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
        if Number == 9:
            cv2.putText(show, "CHERRY", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.imshow('RASPOZNALI', show)
            cv2.waitKey(0)
    print(Number)
#####################################################################################


