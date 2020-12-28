# K Nearest Neighbor
# алгоритм k - ближайших соседей
import random
import numpy as np
import pandas as pd
import pylab as pl
import math
from matplotlib.colors import ListedColormap


# Евклидово расстояние между 2 точками
# def distance (a, b):
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Генерация рандомных данных
def generate_data(number_of_class_element, number_of_classes) -> list:
    data = []
    for class_number in range(number_of_classes):
        # Выбрать случайный центр двумерного гаусса
        center_x, center_y = random.random() * 5.0, random.random() * 5.0
        # Выбрать случайные узлы со среднеквадратичным значением 0,5
        for row_number in range(number_of_class_element):
            data.append([[random.gauss(center_x, 0.5), random.gauss(center_y, 0.5)], class_number])
    return data


# Демо распределения выборки
def show_data(n_classes, n_items_in_class):
    train_data = generate_data(n_items_in_class, n_classes)
    class_colormap = ListedColormap(['#a84747', '#26ff00', '#5881bf'])
    pl.scatter([train_data[i][0][0] for i in range(len(train_data))],
               [train_data[i][0][1] for i in range(len(train_data))],
               c=[train_data[i][1] for i in range(len(train_data))],
               cmap=class_colormap)
    pl.show()


# Разбиение данных на две выборки: обучающую и тестовую.
def split_train_test(data, test_percent):
    train_data = []
    test_data = []
    for row in data:
        if random.random() < test_percent:
            test_data.append(row)
        else:
            train_data.append(row)
    return train_data, test_data


# Реализация классификатора KNN

# Для каждой точки тестовых данных делаем следующее: 1. Рассчитываем расстояние между данными испытаний и каждой
# строкой тренировочных данных с помощью любого из методов (в нашем случае Евклидово расстояние) 2. Теперь,
# основываясь на значении расстояния, сортируем в порядке возрастания. 3. Далее выбираем верхние K строк из
# отсортированного массива. 4. Теперь назначаем класс контрольной точке на основе наиболее часто встречающегося
# класса в этих строках.


def knn_classify(train_data, test_data, k, number_of_classes):
    test_labels = []
    for test_point in test_data:
        # Рассчитать расстояние между test point и всеми train points
        test_dist = [
            [distance(test_point[0],
                      test_point[1],
                      train_data[i][0][0],
                      train_data[i][0][1]),
             train_data[i][1]] for i in range(len(train_data))
        ]

        # Сколько точек каждого класса среди ближайших K
        stat = [0 for _ in range(number_of_classes)]
        for d in sorted(test_dist)[0:k]:
            stat[d[1]] += 1

        # Назначаем класс с наибольшим количеством вхождений среди K ближайших соседей
        test_labels.append(sorted(zip(stat, range(number_of_classes)), reverse=True)[0][1])

    return test_labels


def show_data_on_mesh_after_classify(n_classes, n_items_in_class, k):
    # Создаем сетку узлов, которая охватывает все случаи train
    def generate_test_mesh(input_data):
        x_min = min([input_data[i][0][0] for i in range(len(input_data))]) - 1.0
        x_max = max([input_data[i][0][0] for i in range(len(input_data))]) + 1.0
        y_min = min([input_data[i][0][1] for i in range(len(input_data))]) - 1.0
        y_max = max([input_data[i][0][1] for i in range(len(input_data))]) + 1.0

        h = 0.05
        test_x, test_y = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
        return [test_x, test_y]

    train_data = generate_data(n_items_in_class, n_classes)
    test_mesh = generate_test_mesh(train_data)
    test_mesh_labels = knn_classify(train_data, zip(test_mesh[0].ravel(), test_mesh[1].ravel()), k, n_classes)
    class_colormap = ListedColormap(['#a84747', '#26ff00', '#5881bf'])
    test_colormap = ListedColormap(['#ffae00', '#00ffc3', '#c9f2ca'])

    pl.pcolormesh(test_mesh[0],
                  test_mesh[1],
                  np.asarray(test_mesh_labels).reshape(test_mesh[0].shape),
                  cmap=test_colormap)
    pl.scatter([train_data[i][0][0] for i in range(len(train_data))],
               [train_data[i][0][1] for i in range(len(train_data))],
               c=[train_data[i][1] for i in range(len(train_data))],
               cmap=class_colormap)
    pl.show()


# Оценка, насколько хорошо работает классификатор
def calculate_accuracy(n_classes, n_item_in_class, k, test_percent):
    data = generate_data(n_item_in_class, n_classes)
    train_data, test_data_with_labels = split_train_test(data, test_percent)
    test_data = [test_data_with_labels[i][0] for i in range(len(test_data_with_labels))]
    test_data_labels = knn_classify(train_data, test_data, k, n_classes)
    print("Accuracy: ",
          sum(
              [int(test_data_labels[i] == test_data_with_labels[i][1]) for i in range(len(test_data_with_labels))])
          / float(len(test_data_with_labels)))


# сгенерируем данные, разобьем их на обучающую и тестовую выборку, произведем классификацию
# объектов тестовой выборки и сравним реальное значение класса с полученным в результате классификации


def main():
    n_classes = 4
    n_item_in_class = 30
    test_percent = 0.1
    k = 10
    show_data(n_classes, n_item_in_class)
    show_data_on_mesh_after_classify(n_classes, n_item_in_class, k)

    data = generate_data(n_item_in_class, n_classes)
    train_data, test_data_with_labels = split_train_test(data, test_percent)
    # test_data = [test_data_with_labels[i][0] for i in range(len(test_data_with_labels))]

    calculate_accuracy(n_classes, n_item_in_class, k, test_percent)


if __name__ == '__main__':
    main()
