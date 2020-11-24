from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import time
import numpy as np
import math
'''
    pytonn pymoo包中的NSGA2函数
    效果和速度都比自己写的好
'''


# 判断个体支配关系，用于测试集合
def dominate(p1, p2):
    '''
    :param p1: population1
    :param p2: population2
    :return:
    '''
    less = greater = equal = 0
    f_num = len(p1)
    for i in range(f_num):
        if p1[i] > p2[i]:
            greater += 1
        elif p1[i] == p2[i]:
            equal += 1
        else:
            less += 1
    if less == 0 and equal != f_num:
        return False
    elif greater == 0 and equal != f_num:
        return True
    return None

def compare_ZDT1():
    N = 100
    problem = get_problem("ZDT1")
    algorithm = NSGA2(pop_size=100, elimate_duplicates=False)
    start = time.time()
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 2000),
                   verbose=False)
    end = time.time()
    print('耗时：', end - start, '秒')

    zdt1 = np.loadtxt('ZDT1.txt')
    plt.scatter(zdt1[:, 0], zdt1[:, 1], marker='o', color='green')
    plt.scatter(res.F[:, 0], res.F[:, 1], marker="o", color='red')

    best_solution = zdt1
    # coverage (C-metric) 评价
    number = 0
    for i in range(N):
        dominated_num = 0
        for j in range(len(best_solution)):
            if dominate(best_solution[j], res.F[i]):
                dominated_num += 1
        if dominated_num != 0:
            number += 1
        C_AB = float(number / N)
    print("c-metric %2f" % C_AB)  # 越小越好

    # distance from representatives in the PF (D-metric)
    min_d = 0
    for i in range(len(best_solution)):
        temp = []
        for j in range(N):
            dd = 0
            for k in range(2):
                dd += float((best_solution[i][k] - res.F[j][k]) ** 2)
            temp.append(math.sqrt(dd))
        min_d += np.min(temp)
    D_AP = float(min_d / N)
    print("D-metric: %2f" % D_AP)

    plt.grid(True)
    plt.show()

compare_ZDT1()

'''
    ZDT1的效果比较：
    python包的nsga2(N=100,iter=2k):
    (1)耗时： 17.803385972976685 秒
    c-metric 0.200000
    D-metric: 0.022778
    (2)耗时： 16.81207823753357 秒
    c-metric 0.210000
    D-metric: 0.024440
    (3)耗时： 16.19368290901184 秒
    c-metric 0.140000
    D-metric: 0.022827

    自己实现的nsga2(N=100,iter=2k)：
    (1)循环时间:52.887559秒
    c-metric 0.220000
    D-metric: 0.030732
    
    (2)循环时间:52.235275秒
    c-metric 0.110000
    D-metric: 0.035583
    
'''