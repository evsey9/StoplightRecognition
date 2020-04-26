import numpy as np
import cv2
#stoplight_mask = cv2.imread("stoplight_mask.png")
def one_hot_encode(label):

    """ Функция осуществляет перекодировку текстового входного сигнала
     в массив элементов, соответствующий выходному сигналу

     Входные параметры: текстовая метка

     Выходные параметры: метка ввиде массива

     Пример:

        one_hot_encode("red") должно возвращать: [1, 0, 0]
        one_hot_encode("yellow") должно возвращать: [0, 1, 0]
        one_hot_encode("green") должно возвращать: [0, 0, 1]

     """

    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]

    return one_hot_encoded

# приведение входного изображения к стандартному виду
def standardize_input(image):
    standard_im = image
    """Приведение изображений к стандартному виду. 
    Входные данные: изображение
    Выходные данные: стандартизированное изображений.
    """
    width = 100
    height = 300
    standard_im = cv2.resize(standard_im, (width, height))
    ## TODO: Если вы хотите преобразовать изображение в формат, одинаковый для всех изображений, сделайте это здесь.
    return standard_im

# Определение сигнала светофора по изображению

def predict_label(rgb_image):
    """
     функция определения сигнала светофора по входному изображению

     Входные данные: rgb изображение
     Выходные данные:

    """
    ## TODO: ваша функция распознавания сигнала светофора должна быть здесь.
    labels = ["red", "yellow", "green"]
    predicted_label = "yellow"
    h_min = np.array((0, 70, 70), np.uint8)
    h_max = np.array((255, 255, 255), np.uint8)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)

    #res = cv2.subtract(bgr_image, stoplight_mask)
    #res_hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(hsv_image, h_min, h_max)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #copy = bgr_image.copy()
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #cv2.drawContours(copy, contours, 0, (255, 0, 255), 2)
        x, y, w, h = cv2.boundingRect(contours[0])
        x_center = x + w // 2
        y_center = y + h // 2
        predicted_label = labels[y_center // (bgr_image.shape[0] // 3)]
        #print(y_center)
        #print(predicted_label)
        #cv2.circle(copy, (x_center, y_center), 3, (0, 255, 255), -1)
        #cv2.drawContours(copy, contours, 0, (0, 255, 0), 2)

    #cv2.imshow("img", copy)
    #cv2.imshow("res", res)
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    encoded_label = one_hot_encode(predicted_label)

    return encoded_label
