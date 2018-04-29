# !/usr/bin/python
# -*- coding: utf8 -*-

# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 1

# 1.1 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
	 [0,1,0,0],
	 [0,0,1,0],
	 [0,0,0,1]
]

# 1.2 返回矩阵的行数和列数
def shape(M):
    return len(M),len(M[0])


# 1.3 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = round(M[i][j],decPts)

# 1.4 计算矩阵的转置
# https://docs.python.org/dev/tutorial/controlflow.html#unpacking-argument-lists
# http://python3-cookbook.readthedocs.io/zh_CN/latest/c04/p11_iterate_over_multiple_sequences_simultaneously.html
# 用* -operator写入函数调用以将参数从列表或元组中解开
def transpose(M):
    return [list(col) for col in zip(*M)]

# 1.5 计算矩阵乘法 AB，如果无法相乘则raise ValueError
# http://www.jb51.net/article/68532.htm 
def matxMultiply(A, B):
    try:
        if len(A[0]) != len(B):
            raise ValueError
        return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]
    except ValueError:
        raise ValueError('Two length are not eq.')

# 2.1 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [a + b for a,b in zip(A,b)]

# 2.2 
# 初等行变换
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# 把某行乘以一个非零常数
def scaleRow(M, r, scale):
    if scale != 0:
        for i,row in enumerate(M[r]):
            M[r][i] = M[r][i] * scale
    else:
        raise ValueError('scale cannot be 0.')

# 把某行加上另一行的若干倍
def addScaledRow(M, r1, r2, scale):
    M[r1] = [x + y * scale for x,y in zip(M[r1],M[r2])]

# 2.3 Gaussian Jordan 消元法求解 Ax = b

# 2.3.2 算数步骤详见ipynb

# 2.3.3 实现 Gaussian Jordan 消元法

# https://www.codelast.com/原创全选主元的高斯-约当（gauss-jordan）消元法解线性方/
# https://www.codelast.com/原创全选主元的高斯-约当（gauss-jordan）消元法解线性方/
def gj_Solve(A,b,decPts=4, epsilon = 1.0e-16):
    # 2. 构造增广矩阵
    M = augmentMatrix(A, b)

    maxIndex = 0

    rowCount = len(M)
    columnCount = len(M[0]) - 1

    # 1. 如果行与列数量不等,返回None
    if len(A) != len(A[0]):
        return None
    
    if len(A) != len(b):
        return None

    # 3. RREF
    for c in range(columnCount):
        # 3.1 最大值所在行初始为当前行
        maxRow = c

        # 3.2 找到绝对值的最大值，并设置最大值所在行
        for row in range(c, rowCount):
            if abs(M[row][c]) > maxIndex:
                maxIndex = abs(M[row][c])
                maxRow = row

        # 3.3 最大值若为零返回（奇异矩阵）
        if abs(maxIndex) < epsilon:
            return None      
        else:
            swapRows(M, c, maxRow)
            # 3.5 行内都除以绝对值最大值,使最大值变为1
            if (M[c][c]) >= 1:
                scaleRow(M, c, 1 / M[c][c])

            # 3.4 将绝对值最大值所在行交换到对角线元素所在行（行c） (M, c, maxRow)
            for k in range(c):
                while abs(M[k][c]) >= epsilon:
                    addScaledRow(M, k, c, M[k][c])
            for i in range(c + 1, rowCount):
                while abs(M[i][c]) >= epsilon:
                    addScaledRow(M, i, c, -M[i][c])
    
    # 返回最后一列            
    result = []
    for r in range(rowCount):
        result.append([M[r][-1]])
    return result


# 3 线性回归

# 3.1 随机生成样本点 详见ipynb

# 3.2 拟合一条直线

# 3.2.1 猜测一条直线 详见ipynb

# 3.2.2 计算平均平方误差 (MSE)
def calculateMSE(X,Y,m,b):
    len_of_line = len(X)
    MSE = 0
    for j,num in enumerate(Y):
        MSE += ( (num - m*X[j] - b) **2 )

    MSE = MSE/len_of_line
    
    return MSE


# 3.4 求解  XTXh=XTY
# 差一个gj_Solve就能出结果
def linearRegression(X,Y):
    x = []
    y = []
    for i,r in enumerate(X):
        x.append([r,1])
    for i,r in enumerate(Y):
        y.append([r])
    XT = transpose(x)
    A = matxMultiply(XT, x)
    b = matxMultiply(XT, y)
    result_list = gj_Solve(A, b)
    return result_list[0][0], result_list[1][0]
