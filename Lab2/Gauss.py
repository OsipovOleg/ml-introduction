import numpy as np

np.set_printoptions(precision=3)


# Поиск главного элемента
def MainElement(ECM, i, N, lst):
    Max = 0
    it = i
    # Поиск главного элемента ниже нулевого элемента
    for k in range(i + 1, N):
        Main = abs(ECM[k][i])
        if Main > Max:
            Max = Main
            it = k

    # Перестановка строк
    if Max != 0:
        for k in range(N + 1):
            el = ECM[i][k]
            ECM[i][k] = ECM[it][k]
            ECM[it][k] = el
        return ECM

    Max = 0
    it = i
    # Поиск главного элемента правее нулевого элемента
    for k in range(i + 1, N):
        Main = abs(ECM[i][k])
        if Main > Max:
            Max = Main
            it = k

    # Перестановка столбцов
    if Max != 0:
        lst.append(i)
        lst.append(it)
        for k in range(N):
            el = ECM[k][i]
            ECM[k][i] = ECM[k][it]
            ECM[k][it] = el

    return ECM


# Перестановка элементов вектора результата
def Transpose(X, lst):
    while len(lst) > 0:
        i = lst.pop()
        j = lst.pop()
        el = X[i]
        X[i] = X[j]
        X[j] = el
    return X


# Прямой ход
def DirectMove(M, N, lst):
    # i - шаг алгоритма Гаусса
    # j - строки
    # k - столбцы
    for i in range(N):
        if M[i][i] == 0:
            M = MainElement(M, i, N, lst)

        for j in range(i, N):
            mult = M[j][i]
            for k in range(i, len(M[0])):
                if j == i:
                    M[j][k] *= 1 / mult
                else:
                    M[j][k] -= M[i][k] * mult
    return M


def Gauss(A, b):
    # Extended coefficient matrix
    # Расширенная матрица коэффициентов
    ECM = np.hstack((A, b))
    N = len(b)

    # Список для перестановок столбцов
    lst = []

    # ПРЯМОЙ ХОД
    ECM = DirectMove(ECM, N, lst)

    # ОБРАТНЫЙ ХОД
    X = np.zeros(N)
    for i in range(0, N):
        X[N-1-i] = ECM[N-1-i][N]
        Sum = 0
        for j in range(0, i):
            Sum += X[N-1-j] * ECM[N-1-i][N-1-j]
        X[N-1-i] -= Sum

    X = Transpose(X, lst)

    return X


def main():
    A = np.array([[0.0, 5.0, 9.0], [1.0, 3.0, 1.0], [12.0, 8.0, 1.0]])
    b = np.array([[5.0, 3.0, 0.0]]).T
    print(Gauss(A, b))


if __name__ == '__main__':
    main()
