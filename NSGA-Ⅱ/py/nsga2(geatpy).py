# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import math

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

class ZDT1(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'ZDT1'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 30  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        ObjV1 = Vars[:, 0]
        gx = 1 + 9 * np.sum(Vars[:, 1:], 1) / (self.Dim - 1)
        hx = 1 - np.sqrt(np.abs(ObjV1) / gx)  # 取绝对值是为了避免浮点数精度异常带来的影响
        ObjV2 = gx * hx
        pop.ObjV = np.array([ObjV1, ObjV2]).T  # 把结果赋值给ObjV


if __name__ == "__main__":
    """================================实例化问题对象============================="""
    problem = ZDT1()  # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 2000  # 最大进化代数
    myAlgorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    # NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%f 秒' % myAlgorithm.passTime)

    # 进行评价
    solution = NDSet.ObjV
    zdt1 = np.loadtxt('ZDT1.txt')
    plt.scatter(zdt1[:, 0], zdt1[:, 1], marker='o', color='green')
    plt.scatter(solution[:, 0], solution[:, 1], marker="o", color='red')

    best_solution = zdt1

    N = NIND
    # coverage (C-metric) 评价
    number = 0
    for i in range(N):
        dominated_num = 0
        for j in range(len(best_solution)):
            if dominate(best_solution[j], solution[i]):
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
                dd += float((best_solution[i][k] - solution[j][k]) ** 2)
            temp.append(math.sqrt(dd))
        min_d += np.min(temp)
    D_AP = float(min_d / N)
    print("D-metric: %2f" % D_AP)

    plt.grid(True)
    plt.show()

    '''
        Geatpy 效果最好，Geatpy团队自主研发的超高性能矩阵库进行计算
        用时：2.067473 秒
        c-metric 0.110000
        D-metric: 0.022979
        
        用时：2.262980 秒
        c-metric 0.210000
        D-metric: 0.022555

        
    '''

