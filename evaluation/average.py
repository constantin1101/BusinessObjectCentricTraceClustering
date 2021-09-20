def averageDistance(x, n):
    sum = 0
    for i in range(0, n):
        for j in range(0, n):
            sum = sum + x[i][j]
    average = sum / (n*n)
    return average
