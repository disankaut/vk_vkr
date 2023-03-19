#для вк
import vk_api
import os
import sys
import urllib

#для поиска шевронов
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#для поиска лиц
import face_recognition as fr
import face_recognition
from time import sleep
from os import path
import copy

#для базы
import sqlite3

#для выведения даты и времени в имени каталога
import datetime
from time import time

from PIL import Image
from PIL.ExifTags import TAGS
import datetime
from time import time
import shutil

#для графики
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtGui
from functools import partial
from description.mu import *

#база с данными о фотоаппарате
def db(cwd_host):
    map_model = {}
    conn = sqlite3.connect(cwd_host)
    cursor = conn.cursor()
    query_str = "SELECT soft, model FROM model_soft"
    cursor.execute(query_str)
    records = cursor.fetchall()
    for m in records:
        map_model[m[0]] = m[1]
    conn.close()
    return map_model

def metadata (cwd, cwd_host, tempPrint):
    os.chdir(cwd)
    now = datetime.datetime.now()
    verification_time = now.strftime("%d-%m-%Y %H.%M")
    map_model =db(cwd_host)
    dir_path_users = os.path.join('OutputReport')
    if not os.path.exists(dir_path_users):
        os.makedirs(dir_path_users)
        # print (dir_path_users)
    dir_path_users = os.path.join('OutputReport\\FakePhotos')
    if not os.path.exists(dir_path_users):
        os.makedirs(dir_path_users)

    #создали каталог при запуске программы cо временем проверки, если его нет
    timeDir_path = os.path.join('OutputReport\\FakePhotos\\'+verification_time)

    if not os.path.exists(timeDir_path):
        os.makedirs(timeDir_path)

    sys.stdout = open(timeDir_path + "\\Отчет.txt", "w")

    os.chdir(os.path.join(cwd, "InputImages"))
    files = os.listdir()
    if len(files) == 0:
        print("Каталог с изображениями пуст.")
        sys.stdout.close()
        os.chdir(cwd)
        return

    for file in files:
        try:
            flag_soft = False
            flag_model = False
            flag_time = False
            tag_model = ""
            tag_soft  = ""
            tag_datetimeStart = ""
            tag_datetimeStop = ""
            image = Image.open(file)
            if image._getexif() == None:
                print(f"На фотоизображении {file} отсутствуют метаданные. Фотоизображение сфальсифицировано.\n\n")
                shutil.copy(os.path.join(cwd + "\\InputImages", file), os.path.join(cwd+"\\"+timeDir_path))
                continue
            else:
                for tag, value in image._getexif().items():
                    tag_name = TAGS.get(tag)

                    if tag_name == "Model":
                        tag_model = value
                        for d in map_model.values():
                            if str(value) == d:
                                flag_model = True

                    if tag_name == "Software":
                        tag_soft = value
                        for d in map_model.keys():
                            if str(value) == d:
                                flag_soft = True
                        if flag_soft:
                            if map_model[str(value)] != tag_model:
                                flag_soft = False

                    if tag_name == "DateTime":
                        tag_datetimeStart = str(value)

                    if tag_name == "DateTimeOriginal":
                        tag_datetimeStop = str(value)

            if tag_datetimeStart == tag_datetimeStop:
                flag_time = True
            else:
                flag_time = False

            if flag_model==False or flag_soft==False or flag_time==False:
                print (f"Фотоизображение {file}  сфальсифицировано.")
                if flag_model==False:
                    print ("\tМодель фотоаппарата не соответстсвует зарегестрированной.")
                    print (f"\t\tModel - {tag_model}")
                if flag_soft == False:
                    if tag_soft!="":
                        print ("\n\tСерийный номер/программное обеспечение фотоаппарата не зарегистрирован(о).")
                        print (f"\t\tSoftware - {tag_soft}")
                    else:
                        print ("\n\tСерийный номер/программное обеспечение фотоаппарата удалено из метаданных фотоизображения.")
                if (flag_time == False) and (tag_datetimeStart!="" or tag_datetimeStop!=""):
                    print ("\n\tДата/время съемки и редактирования отличаются.")
                    print (f"\t\tДата/время съемки - {tag_datetimeStop}")
                    print (f"\t\tДата/время редактирования - {tag_datetimeStart}\n\n")

                shutil.copy(os.path.join(cwd + "\\InputImages", file), os.path.join(cwd + "\\" + timeDir_path))

        except IOError:
            print("Некорректный формат файлов в каталоге.")
        image.close()

    sys.stdout.close()
    sys.stdout = tempPrint

#ищем информацию о найденной фотографии в базе ИЛИ записываем в базу фальсификации
def SELECTstandard(name):
    name+=".jpg"
    conn = sqlite3.connect(r"rec\military.db")
    cursor = conn.cursor()
    query_str = "SELECT title, fio, post, division, number FROM 'standard' WHERE photo=" +  "\"" +  str(name)+  "\""
    cursor.execute(query_str)
    model = cursor.fetchall()
    for m in model:
        return m

#записывем в таблицу нарушителя информацию о нарушителе (ЗГТ)
def INSERTviolation(informationList, name, link):
    conn = sqlite3.connect(r"rec\military.db")
    cursor = conn.cursor()
    title = informationList[0]
    fio = informationList[1]
    post = informationList[2]
    division = informationList[3]
    number = informationList[4]
    insSqlQueryStr_template = "INSERT INTO 'violation' ('title' , 'fio', 'post', 'division', 'number', 'photo', 'link') VALUES" \
                              " ({titleS}, {fioS}, {postS}, {divisionS}, {numberS}, {photoS}, {linkS})"
    insSqlQueryStr = insSqlQueryStr_template.format(titleS=title,
                                                    fioS=fio,
                                                    postS = post,
                                                    divisionS=division,
                                                    numberS=number,
                                                    photoS = name,
                                                    linkS = link)
    cursor.execute(insSqlQueryStr)
    conn.commit()

#записываем в таблицу фотофальсификации информации о нарушителе (фалисификаторе)
def INSERTphotofalsification(img_answer_path, link):
    conn = sqlite3.connect(r"rec\military.db")
    cursor = conn.cursor()
    insSqlQueryStr_template = "INSERT INTO 'photofalsification' ('photo' , 'link') VALUES ({nameS}, {linkS})"
    insSqlQueryStr = insSqlQueryStr_template.format(nameS = img_answer_path,
                                                    linkS = link )
    cursor.execute()
    conn.commit()

#получили все id пользователей сообщества
def get_users_group(vk, group_id):
  users = vk.groups.getMembers(group_id=group_id)
  return users

#вспомогательлная функция получения id группы из ссылки
def take_the_id_from_the_link(vk, link):
  a_l=""
  b_l=""
  for i in link:
    if(a_l != "https://vk.com/"):
      a_l+=i
    else:
      b_l+=i

  group_id = vk.utils.resolveScreenName(screen_name= b_l)['object_id']
  return group_id

#Удаление пустых папок в директории
def del_empty_dirs(path):
    for d in os.listdir(path):
        a = os.path.join(path, d)
        if os.path.isdir(a):
            del_empty_dirs(a)
            if not os.listdir(a):
                os.rmdir(a)
                print(a, 'удалена')

# набор функций по определению лиц
#1
def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}
    files = os.listdir(cwdGLOBAL+r"\rec\FaceDataImg")
    for f in files:
        if f.endswith(".jpg") or f.endswith(".png"):
            face = fr.load_image_file("rec\\FaceDataImg\\" + f)

            encoding = fr.face_encodings(face)[0]

            encoded[f.split(".")[0]] = encoding
    return encoded
#2
def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("rec\\FaceDataImg\\" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding
#3
def classify_face(img_path,newpath_user, lst_person):
    #print(lst_person)
    #print ('img_path = ', img_path)

    #сначала для нераспознанных
    #print ("+")
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())
    img = cv2.imread(img_path, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
    face_names = []
    name = "Unknown"
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            if name == "Unknown":
                for kp in lst_person:
                    xw = kp[0] + kp[2]
                    yh = kp[1] + kp[3]
                    #доп условие, что если есть шеврон в рамке человека

                    if (left <= xw and left >= kp[0]) and (top <= yh and top >= kp[1]) and(
                            right <= xw and right >= kp[0]) and (bottom<=yh and bottom >= kp[1]) and kp[4] == True:
                        #print("++")
                        cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (0, 0, 255), 2)
                        text = "unknown"
                        # вычисляем ширину и высоту текста, чтобы определить прозрачные поля в качестве фона текста
                        (text_width, text_height) = \
                        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                        text_offset_x = left-20
                        text_offset_y = bottom + 45
                        box_coords = (
                        (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = img.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(0, 0, 255), thickness=cv2.FILLED)
                        # добавить непрозрачность (прозрачность поля)
                        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                        cv2.putText(img, text, (left-20, bottom +45), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(255, 255, 255), thickness=1)
            #будем проверять name  в базе

        #вывод результатов
        #print(name)
        if name == 'Unknown':
            img_name = os.path.basename(img_path)
            user_path = newpath_user + "\\" +'Unknown'
            if not os.path.exists(user_path):
                os.makedirs(user_path)

            out_file = open(user_path + '\\Unknown.txt', "w")
            link = os.path.basename(os.path.normpath(newpath_user))
            out_file_write_str = "Данных военнослужащих, отмеченных 'Unknown', нет в базе данных -> фотографии сфальсифицированы\n" \
                                 "Ссылка на интернет-страницу пользователя, который выложил фотографии" \
                                 + ":\n"+ "https://vk.com/id"+link+"\n\n"
            out_file.write(out_file_write_str)
            out_file.close()
            cv2.imwrite(user_path + '\\' +img_name, img)
            #os.remove(img_path)                                     # удалили старую


    #затем на распознанные
    #print ("+")
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img1 = cv2.imread(img_path, 1)
    im_b = copy.copy(img1)
    face_locations = face_recognition.face_locations(img1)
    unknown_face_encodings = face_recognition.face_encodings(img1, face_locations)
    face_names = []
    #print (face_locations)
    for face_encoding in unknown_face_encodings:
        #print ("+")
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        #face_names = [name]
        #print (face_locations)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if name!="Unknown":
                cv2.rectangle(im_b, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
                text = name
                # вычисляем ширину и высоту текста, чтобы определить прозрачные поля в качестве фона текста
                (text_width, text_height) = \
                    cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
                text_offset_x = left - 20
                text_offset_y = bottom + 45
                box_coords = (
                    (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = im_b.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=(255, 0, 0), thickness=cv2.FILLED)
                # добавить непрозрачность (прозрачность поля)
                im_b = cv2.addWeighted(overlay, 0.6, im_b, 0.4, 0)
                cv2.putText(im_b, text, (left - 20, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255), thickness=1)
                #cv2.rectangle(im_b, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
                #cv2.putText(im_b, name, (left, bottom + 15), 1, 1.5, (255, 255, 255), 2)
                img_name = os.path.basename(img_path)
                user_path = str(newpath_user)+"\\" + name
                #print (user_path)
                #создали папку с именем нарушителя
                if not os.path.exists(user_path):
                    os.makedirs(user_path)
                cv2.imwrite(user_path + "\\" + img_name, im_b)  # сохранили туда одну из фоток
                im_b = copy.copy(img1)
        if name != "Unknown":
            #создали текстовый файл
            out_file = open(user_path + "\\" + name + ".txt", "a+")
            #out_file.close()
            lst_infUser = SELECTstandard(name)          #name =                      Tarakanow                  - имя папки
            #print(lst_infUser)
            out_file.write("На фотографии " + img_name + " получена информация:\n")
            out_file.write("\tВоинское звание: " + lst_infUser[0] +"\n")
            out_file.write("\tФамилия Имя Отчество:  " + lst_infUser[1] +"\n")
            out_file.write("\tДолжность: " + lst_infUser[2] +"\n")
            out_file.write("\tПодразделение: " + lst_infUser[3] +"\n")
            out_file.write("\tЛичный номер: " + lst_infUser[4] +"\n")
            out_file.write("\n")
            out_file.close()

    if name == 'Unknown':
        img_name = os.path.basename(img_path)
        user_path = newpath_user + "\\" + 'Unknown'
        if not os.path.exists(user_path):
            os.makedirs(user_path)
        cv2.imwrite(user_path + "\\" + img_name, img)  # сохранили туда одну из фоток
        out_file = open(user_path + '\\Unknown.txt', "w")
        link = os.path.basename(os.path.normpath(newpath_user))
        out_file_write_str = "Данных военнослужащих ВКА им. А.Ф.Можайского нет в базе данных.\n" \
                             "Необходима экспертиза на фальсификацию данных фотоизображений.\n" \
                             "Ссылка на интернет-страницу пользователя, который выложил фотографии" \
                             + ":\n" + "https://vk.com/id" + link + "\n\n"
        out_file.write(out_file_write_str)
        out_file.close()
    os.remove(img_path)

#файлы весов и конфиг, полный путь до фото
# Функция поиска шеврона на фото                ОТРАБАТЫВАЕТ!
def search_vka(file_weights, file_cfg, img_path, newpath_user, lst_person):
    net = cv2.dnn.readNet(file_weights,file_cfg)
    classes = ['vka']
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 1))
    img = cv2.imread(img_path)
    #cv2.imshow("qe", img)
    #cv2.waitKey(5000)
    #sleep(7)
    img_b = img
    hight, width, _ = img.shape
    #print (hight, width, _)
    img = img_b
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence >= 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.05)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    # получение координат и рамки на фотографию + вероятность
    countObject = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            for kp in lst_person:
                    xw = kp[0]+kp[2]
                    yh = kp[1]+kp[3]
                    if (x<=xw and x>=kp[0]) and (y<=yh and y>=kp[1]) and ((x+w) >= kp[0]  and (x+w)<= xw ) and ((y+h) >= kp[1] and (y+h) <= yh):
                        #print ("+")
                        label = str(classes[class_ids[i]])
                        confidence = str(round(confidences[i], 2))
                        color = [0, 252, 252]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                        temp = True
                        kp[4] = temp

    if (float(confidence) > 0.3):
        cv2.imwrite(img_path, img)              #заменяет старые фото на распознанные
        #os.remove(img_path)
        cv2.destroyAllWindows()

        classify_face(img_path, newpath_user, lst_person)

    else:
        #print ("No")
        os.remove(img_path)  #удалили то, что не является шевроном
        cv2.destroyAllWindows()

#функция поиска человека на фото +шеврон
def searchPerson(file_cfg, file_weights, img_path, newpath_user):
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    lst_person = []
    # конфигурация нейронной сети
    config_path = "rec\\weights\yolov3.cfg"
    # файл весов сети YOLO
    weights_path = "rec\\weights\yolov3.weights"
    # weights_path = "weights/yolov3-tiny.weights"

    # загрузка всех меток классов (объектов)
    labels = open("rec\\weights\\coco.names").read().strip().split("\n")
    # генерируем цвета для каждого объекта и последующего построения
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # загружаем сеть YOLO
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    #path_name = "D:\diplWork\Photo\predict\pred35.jpg"
    image = cv2.imread(img_path)
    file_name = os.path.basename(img_path)
    filename, ext = file_name.split(".")

    h, w = image.shape[:2]
    # создать 4D blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # устанавливает blob как вход сети
    net.setInput(blob)
    # получаем имена всех слоев
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # прямая связь (вывод) и получение выхода сети
    # измерение времени для обработки в секундах
    #start = time.perf_counter()
    layer_outputs = net.forward(ln)
    #time_took = time.perf_counter() - start

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # перебираем каждый из выходов слоя
    for output in layer_outputs:
        # перебираем каждое обнаружение объекта
        for detection in output:
            # извлекаем идентификатор класса (метку) и достоверность (как вероятность)
            # обнаружение текущего объекта
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # отбросьте слабые прогнозы, убедившись, что обнаруженные
            # вероятность больше минимальной вероятности
            if confidence > CONFIDENCE:
                # масштабируем координаты ограничивающего прямоугольника относительно
                # размер изображения, учитывая, что YOLO на самом деле
                # возвращает центральные координаты (x, y) ограничивающего
                # поля, за которым следуют ширина и высота поля
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # используем центральные координаты (x, y) для получения вершины и
                # и левый угол ограничительной рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # обновить наш список координат ограничивающего прямоугольника, достоверности,
                # и идентификаторы класса
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # выполнить не максимальное подавление с учетом оценок, определенных ранее
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    # убедитесь, что обнаружен хотя бы один объект

    if len(idxs) > 0:
        # перебираем сохраняемые индексы
        for i in idxs.flatten():
            if labels[class_ids[i]] == 'person':
                # извлекаем координаты ограничивающего прямоугольника
                x, y = boxes[i][0]-30, boxes[i][1]-30
                w, h = boxes[i][2]+30, boxes[i][3]+30
                buf = [x, y, w, h, False]
                lst_person.append(buf)

                # рисуем прямоугольник ограничивающей рамки и подписываем на изображении
                color = [int(c) for c in colors[class_ids[0]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                # вычисляем ширину и высоту текста, чтобы определить прозрачные поля в качестве фона текста
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # добавить непрозрачность (прозрачность поля)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # теперь поместим текст (метка: доверие%)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)


    #cv2.imwrite("WORK\\" + filename + "_yolo3." + ext, image)
    cv2.imwrite(img_path, image)
    search_vka(file_weights,file_cfg,img_path,newpath_user, lst_person)

# функция скачивания ОТРАБАТЫВАЕТ!
def searchFromVk_vkaFace(file_cfg, file_weights, vk, group_id, timeDir_path):
    users = get_users_group(vk, group_id)  # список id пользователей
    # выкачивание фото из вк
    for user in users['items']:
        # поменять путь на локальный
        newpath_user = os.path.join(timeDir_path, str(user))
        #print (newpath_user)
        if not os.path.exists(newpath_user):
            os.makedirs(newpath_user)
        # посылаем запрос к VK API, count свой, но не более 200
        try:
            response = vk.photos.getAll(owner_id=int(user), count=200)
            #print (response)
        except Exception as e:
            print("Иcключение по фото", e)
            continue

        for ind_res in range(len(response["items"])):
            #print (response['items'][ind_res]['sizes'])
            # берём ссылку на максимальный размер фотографии
            try:

                photo_url = str(
                    response["items"][ind_res]["sizes"][len(response["items"][ind_res]["sizes"]) - 1]["url"])
                #print(newpath_users + '\\'+ str(user) + '\\' + str(response["items"][ind_res]['id']) + '.jpg')
                # скачиваем фото в папку с ID пользователя
                urllib.request.urlretrieve(photo_url, timeDir_path + '\\' + str(user) + '\\' + str(
                    response["items"][ind_res]['id']) + '.jpg')
                # адрес в памяти на фотку
                img_path = timeDir_path + '\\' + str(user) + '\\' + str(response["items"][ind_res]['id']) + '.jpg'
                img_answer_path = str(response["items"][ind_res]['id']) + '.jpg'

                #img = search_vka(file_weights, file_cfg, img_path, newpath_user)
                searchPerson(file_cfg, file_weights, img_path, newpath_user)

            except Exception as e:
                print(e)
                continue

#обобщенная функция для анализа фальсификации события в соц сети
def VK_API(cwd, file_cfg, file_weights, token, link):
    os.chdir(cwd)
    # блок с анализом фальсификации события в соц сети
    session = vk_api.VkApi(token=token)
    vk = session.get_api()
    now = datetime.datetime.now()  # получили дату и время
    verification_time = now.strftime("%d-%m-%Y %H.%M")

    group_id = take_the_id_from_the_link(vk, link)  # получили id группы из ссылки

    # создали каталог при запуске программы, если его нет
    dir_path_users = os.path.join('OutputReport')
    if not os.path.exists(dir_path_users):
        os.makedirs(dir_path_users)
        # print (dir_path_users)
    dir_path_users = os.path.join('OutputReport\\FakeEvent')
    if not os.path.exists(dir_path_users):
        os.makedirs(dir_path_users)

    # создали каталог при запуске программы cо временем проверки, если его нет
    timeDir_path = os.path.join('OutputReport\\FakeEvent\\' + verification_time + " club" + str(group_id))
    if not os.path.exists(timeDir_path):
        os.makedirs(timeDir_path)
        # print(timeDir_path)

    searchFromVk_vkaFace(file_cfg, file_weights, vk, group_id, timeDir_path)  # выкачиваем все фотки профилей и ищем
    del_empty_dirs(timeDir_path)  # удаляем пустые каталоги
