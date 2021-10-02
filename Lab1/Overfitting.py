import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3)


# Средне квадратическая ошибка
def MSE(y_true, y):
    Q = 0
    for i in range(len(y)):
        Q += (y_true[i] - y[i]) * (y_true[i] - y[i])
    return Q / len(y)


def middle(arr):
    arr_sum = 0
    for i in range(len(arr)):
        arr_sum += arr[i]
    return arr_sum / len(arr)


def denumenatorR2(y):
    sumDenR2 = 0
    y_mid = middle(y)
    for i in range(len(y)):
        sumDenR2 += (y[i] - y_mid) * (y[i] - y_mid)
    return sumDenR2


def R2(y_true, y):
    num = MSE(y_true, y) * len(y)
    den = denumenatorR2(y_true)
    return 1 - num / den


def graph(x, y, x_graph, y_graph_model, y_graph_sin):
    plt.scatter(x, y, c='blue')
    plt.plot(x_graph, y_graph_model, c='red')
    plt.plot(x_graph, y_graph_sin, c='green')
    plt.show()


def main():
    n = 10
    d = 25

    D = np.linspace(0, d, d+1)
    N = np.linspace(15, 100 * n, n+1)

    # Для конкретного задания выбрать n или d
    MSE_train = np.zeros(shape=(n+1, 1))
    MSE_test = np.zeros(shape=(n+1, 1))

    # Для конкретного задания выбрать n или d
    for i in range(n+1):

        # Тренировочный датасет
        x = np.linspace(0, 7, int(N[i]))    # Для конкретного задания нужно зафиксировать N
        y_true = 2 * np.sin(x)

        # Выборка
        y = y_true + np.random.normal(0, 1, len(y_true))

        # Разбиение выборки
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Создание модели на основе тренировончого датасета
        # Модель строится с помощью функций из библиотек
        model = np.poly1d(np.polyfit(x_train, y_train, D[5]))   # Для конкретного задания нужно зафиксировать N

        # Предсказанные значения на выборках
        y_pred_train = model(x_train)
        y_pred_test = model(x_test)

        # Истинные значения на выборках
        y_train_true = 2 * np.sin(x_train)
        y_test_true = 2 * np.sin(x_test)

        # Эмпирический риск
        Q_train = MSE(y_train_true, y_pred_train)
        Q_test = MSE(y_test_true, y_pred_test)
        print('Q_train = ' + str(Q_train))
        print('Q_test = ' + str(Q_test))

        MSE_train[i] = Q_train
        MSE_test[i] = Q_test

        # Коэффициент детерминации
        R2_train = R2(y_train_true, y_pred_train)
        R2_test = R2(y_test_true, y_pred_test)
        print('R2_train = ' + str(R2_train))
        print('R2_test = ' + str(R2_test))

        # Массивы для отрисовки графиков на выборках
        x1_graph = np.linspace(np.amin(x_train), np.amax(x_train), 100)
        y1_graph_model = model(x1_graph)
        y1_graph_sin = 2 * np.sin(x1_graph)

        x2_graph = np.linspace(np.amin(x_test), np.amax(x_test), 100)
        y2_graph_model = model(x2_graph)
        y2_graph_sin = 2 * np.sin(x2_graph)

        # Отрисовка графиков
        #graph(x_train, y_train, x1_graph, y1_graph_model, y1_graph_sin)
        #graph(x_train, y_train, x2_graph, y2_graph_model, y2_graph_sin)

    # Для конкретного задания выбрать N или D
    plt.plot(N, MSE_train, c='green')
    plt.plot(N, MSE_test, c='red')
    plt.show()


if __name__ == '__main__':
    main()
