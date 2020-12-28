import cv2
import matplotlib.pyplot as plt

face_utilsfont = cv2.FONT_HERSHEY_SIMPLEX

cascPath = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
eyePath = "opencv/data/haarcascades/haarcascade_eye.xml"
smilePath = "opencv/data/haarcascades/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)


def detect_faces():
    return faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def show_img(img):
    plt.figure(figsize=(16, 12))
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':

    img_path = 'faces.jpg'

    # Загрузка изображения
    gray = cv2.imread(img_path, 0)

    print("Изначальное фото переведенное в черно-белое:\n")
    show_img(gray)

    faces = detect_faces()

    # Проход по списку лиц
    for (x, y, w, h) in faces:
        # Отрисовка прямоугольника по полученным координатам лица на изображении
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)

    print("Фото с выделенными лицами:\n")
    show_img(gray)
