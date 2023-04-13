import numpy as np


def dtw(p: list, q: list):
    # create empty cost matrix
    cost_m = [[0 for _ in range(len(q))] for _ in range(len(p))]

    # заполним матрицу
    cost_m[0][0] = abs(p[0] - q[0])
    for col in range(0, len(q)):

        if col != 0:
            cost_m[0][col] = abs(p[0] - q[col]) + cost_m[0][col - 1]

        for row in range(1, len(p)):
            if col == 0:
                cost_m[row][col] = abs(p[row] - q[col]) + cost_m[row - 1][col]
                continue

            cost_m[row][col] = \
                abs(p[row] - q[col]) + \
                min(cost_m[row - 1][col - 1], cost_m[row - 1][col], cost_m[row][col - 1])
            
    # wraping path identification
    row, col = len(p) - 1, len(q) - 1
    d = []
    d.append(cost_m[row][col])

    while row > 0 or col > 0:
        neighbors = [float("inf"), float("inf"), float("inf")]
        if row > 0 and col > 0:
            neighbors[2] = cost_m[row - 1][col - 1]
        if row > 0:
            neighbors[1] = cost_m[row - 1][col]
        if col > 0:
            neighbors[0] = cost_m[row][col - 1]
        idx = np.argmin(neighbors) + 1
        step = (idx // 2, idx % 2)
        row, col = row - step[0], col - step[1]
        d.append (cost_m[row][col])     

    return sum(d) / len(d), d, cost_m
