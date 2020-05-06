def identity(N):
    M = [[0 for x in range(N)] for y in range(N)]
    for i in range(0, N):
        M[i][i] = 1
    return M


def complement(A, p, q, n, N):
    i = 0
    j = 0
    temp = []
    k = -1
    for row in range(0, n):
        if not (row == p):
            temp.append([])
            k += 1
        for col in range(0, n):
            if not (row == p or col == q):
                temp[k].append(A[row][col])

    return temp


def determinant(A, n, N):
    d = 0
    if n == 1:
        return A[0][0]
    sign = 1
    for f in range(0, n):
        compl = complement(A, 0, f, n, N)
        d = d + sign * A[0][f] * determinant(compl, n - 1, N)
        sign = -sign

    return d


def adjunct(A, n, N):
    adj = [[0 for _ in range(N)] for _ in range(N)]
    if N == 1:
        adj[0][0] = 1
        return

    sign = 1
    for i in range(0, N):
        for j in range(0, N):
            compl = complement(A, i, j, N, N)
            if (i + j) % 2 != 0:
                sign = -1
            else:
                sign = 1
            adj[j][i] = sign * determinant(compl, N - 1, N)

    return adj


def invertmatrix(A, n, N):
    det = determinant(A, N, N)
    inv = [[0 for _ in range(N)] for _ in range(N)]
    adj = adjunct(A, n, N)
    for i in range(0, N):
        for j in range(0, N):
            inv[i][j] = adj[i][j] / det

    return inv


def my_invert(A):
    inv = invertmatrix(A, len(A), len(A))
    return inv

