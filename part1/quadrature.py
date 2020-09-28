import numpy as np
quadrature_table_1D = {
    1: [[0 , 2]],
    2: [[- 1 / np.sqrt(3), 1],
        [1 / np.sqrt(3), 1]],
    3: [[- np.sqrt(3 / 5), 5 / 9],
        [0, 8/9],
        [ np.sqrt(3 / 5), 5 / 9]],
    4: [[- np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36],
        [- np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
        [+ np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), (18 + np.sqrt(30)) / 36],
        [+ np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), (18 - np.sqrt(30)) / 36]]
}
def quadrature1D(a, b, Nq, g):
    integral = 0
    for row in quadrature_table_1D[Nq]:
        integral += g(row[0]*(b-a)/2 + (b+a) / 2)*row[1]
    return integral*(b-a) / 2

for Nq in range(1,5):
    a = 1
    b = 2
    print(quadrature1D(a, b, Nq, np.exp) - np.exp(b) + np.exp(a))


