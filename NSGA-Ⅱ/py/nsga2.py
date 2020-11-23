'''
    程序功能：python实现NSGA2算法，测试函数为ZDT1、2、3、4、6，DTZ1、2
    参考论文：
    1. A fast and Elitist Multiobjective Genetic Algorithm: NSGA-Ⅱ
    2020.11.17 DuYuming
'''
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from mpl_toolkits.mplot3d import Axes3D


# 选择测试函数
def chooseFun(fun_name):
    if fun_name == 'ZDT1':
        f_num = 2
        x_num = 30
        zdt1 = np.loadtxt('ZDT1.txt')
        plt.scatter(zdt1[:, 0], zdt1[:, 1], marker='o', color='green')
        best_solution = zdt1
    elif fun_name == 'ZDT2':
        f_num = 2
        x_num = 30
        zdt2 = np.loadtxt('ZDT2.txt')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        best_solution = zdt2
    elif fun_name == 'ZDT3':
        f_num = 2
        x_num = 30
        zdt3 = np.loadtxt('ZDT3.txt')
        plt.scatter(zdt3[:, 0], zdt3[:, 1], marker='o', color='green', s=40)
        best_solution = zdt3
    elif fun_name == 'ZDT4':
        f_num = 2
        x_num = 10
        zdt4 = np.loadtxt('ZDT4.txt')
        plt.scatter(zdt4[:, 0], zdt4[:, 1], marker='o', color='green', s=40)
        best_solution = zdt4
    elif fun_name == 'ZDT6':
        f_num = 2
        x_num = 10
        zdt6 = np.loadtxt('ZDT6.txt')
        plt.scatter(zdt6[:, 0], zdt6[:, 1], marker='o', color='green', s=40)
        best_solution = zdt6
    elif fun_name == 'DTLZ1':
        f_num = 3
        x_num = 10
        dtlz1 = np.loadtxt('DTLZ1.txt')
        best_solution = dtlz1
    elif fun_name == 'DTLZ2':
        f_num = 3
        x_num = 10
        dtlz2 = np.loadtxt('DTLZ2.txt')
        best_solution = dtlz2
    if fun_name == 'ZDT4':
        x_min = np.array([[0, -5, -5, -5, -5, -5, -5, -5, -5, -5]], dtype=float)  # 决策变量的最小值
        x_max = np.array([[1, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=float)  # 决策变量的最大值
    else:
        x_min = np.zeros((1, x_num))
        x_max = np.ones((1, x_num))

    return f_num, x_num, x_min, x_max, best_solution


class NSGA2():
    def __init__(self, fun_name, x_num, N, f_num, max_iter, pc, pm, yita1, yita2, x_min, x_max):
        self.fun_name = fun_name
        self.f_num = f_num
        self.x_num = x_num
        self.N = N
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm
        self.yita1 = yita1
        self.yita2 = yita2
        self.population = []
        self.x_min = x_min
        self.x_max = x_max

    # 初始化 population
    def init_population(self):
        N = self.N
        x_num = self.x_num
        x_min = self.x_min
        x_max = self.x_max
        for i in range(N):
            chromo = [0 for _ in range(x_num)]
            for j in range(x_num):
                chromo[j] = x_min[0, j] + (x_max[0, j] - x_min[0, j]) * random.random()
            self.population.append(Individual(chromo))

    # 非支配排序
    def non_dominated_sort(self, population):
        N = len(population)
        f_num = self.f_num
        pareto_rank = 1
        rank_indi_dict = {}  # rank和indi的对应关系（一对多，计算拥挤度的时候能方便用到）

        rank_indi_dict[pareto_rank] = []
        for i in range(N):
            for j in range(i + 1, N):
                is_dominate = population[i].dominate(population[j])
                if is_dominate:  # 支配
                    population[i].dominate_set.append(population[j])
                    population[j].dominated_num += 1
                elif is_dominate == False:  # 被支配
                    population[j].dominate_set.append(population[i])
                    population[i].dominated_num += 1

        for i in range(N):
            if population[i].dominated_num == 0:
                population[i].pareto_rank = 1
                rank_indi_dict[pareto_rank].append(population[i])  # 直接加入对象，而不是加入下标

        # 依次构建剩下层次
        while len(rank_indi_dict[pareto_rank]) != 0:
            rank_indi_dict[pareto_rank + 1] = []
            for indi in rank_indi_dict[pareto_rank]:
                for cur_dominated_indi in indi.dominate_set:
                    cur_dominated_indi.dominated_num -= 1
                    if cur_dominated_indi.dominated_num == 0:  # 加上rank，然后加入rank-indi字典
                        cur_dominated_indi.pareto_rank = pareto_rank + 1
                        rank_indi_dict[pareto_rank + 1].append(cur_dominated_indi)
                    # if cur_dominated_indi.dominated_num < 0:
                    #     print("err")
            pareto_rank += 1
        return rank_indi_dict

    # 对种群进行拥挤度计算
    def crowding_distance_sort(self, rank_indi_dict):
        '''
        :param rank_indi_dict: 每一pareto_rank层的个体
        :return: ranked_population: crowd计算完的种群
        '''
        f_num = self.f_num
        ranked_population = []
        # 根据pareto_rank排序
        for pareto_rank in rank_indi_dict:  # 遍历从1开始的层数
            if len(rank_indi_dict[pareto_rank]) == 0:  # rank_indi_dict最后一层为空
                continue
            temp = rank_indi_dict[pareto_rank]
            for k in range(f_num):
                temp = sorted(temp, key=lambda Individual: Individual.f[k])
                temp[0].crowd = float('inf')  # 边界为inf
                temp[-1].crowd = float('inf')
                fmin = temp[0].f[k]
                fmax = temp[-1].f[k]
                # 计算中间的
                for i in range(1, len(temp) - 1):
                    # if temp[i - 1].f == temp[i].f:
                    #     temp[i].crowd = 0
                    #     pass
                    pre_f = temp[i - 1].f[k]
                    next_f = temp[i + 1].f[k]
                    if fmax == fmin:  # 所有子代的此函数值全一样
                        temp[i].crowd = float('inf')
                    else:
                        temp[i].crowd += float((next_f - pre_f) / (fmax - fmin)) / f_num

            ranked_population += temp  # 数组的连接
        return ranked_population

    # 锦标赛选择
    def tournament_selection2(self, population):
        '''
        1. 从N个个体中随机选择k个个体（这里选2）
        2. 根据每个个体的适应度，选择其中适应度最好的进入
        '''
        N = self.N
        tour_size = 2
        population_parent = []
        for i in range(N):  # N此选择，每次二选一
            tempI = int(random.random() * N)
            tempJ = int(random.random() * N)
            while tempJ == tempI:
                tempJ = int(random.random() * N)
            if population[tempI].betterThan(population[tempJ]):
                new_indi = Individual(population[tempI].x)
                population_parent.append(new_indi)
            else:
                new_indi = Individual(population[tempJ].x)
                population_parent.append(new_indi)
        return population_parent

    # 模拟二进制交叉和多项式变异(这部分的数学公式是从网上找的)
    def cross_mutation(self, population_parent):
        x_num = self.x_num
        x_min = self.x_min
        x_max = self.x_max
        pc = self.pc
        pm = self.pm
        yita1 = self.yita1
        yita2 = self.yita2
        N = len(population_parent)

        population_offspring = []
        # 随机选择两个父代个体,进行交叉、变异操作
        for i in range(int(N / 2)):
            tempI = int(N * random.random())
            tempJ = int(N * random.random())
            while tempI == tempJ:
                tempJ = int(N * random.random())
            new_indi1 = Individual(population_parent[tempI].x)
            new_indi2 = Individual(population_parent[tempJ].x)
            off1 = new_indi1
            off2 = new_indi2

            # 交叉概率和变异概率
            temp_pc = random.random()
            temp_pm = random.random()
            # 交叉操作
            if temp_pc < pc:
                # 初始化子代种群
                off1x = []
                off2x = []
                # 模拟二进制交叉(x[]的每一个位置都可能交叉或变异)
                for j in range(x_num):
                    u1 = random.random()
                    if u1 <= 0.5:
                        gama = float((2 * u1) ** (1 / (yita1 + 1)))
                    else:
                        gama = float((1 / (2 * (1 - u1))) ** (1 / (yita1 + 1)))
                    off1j = float(0.5 * ((1 + gama) * off1.x[j] + (1 - gama) * off2.x[j]))
                    off2j = float(0.5 * ((1 - gama) * off1.x[j] + (1 + gama) * off2.x[j]))
                    # 保证在定义域内
                    if off1j > x_max[0][j]:
                        off1j = x_max[0][j]
                    elif off1j < x_min[0][j]:
                        off1j = x_min[0][j]
                    if off2j > x_max[0][j]:
                        off2j = x_max[0][j]
                    elif off2j < x_min[0][j]:
                        off2j = x_min[0][j]
                    off1x.append(off1j)
                    off2x.append(off2j)
                # 这时会更新子代F
                off1 = Individual(off1x)
                off2 = Individual(off2x)

            if temp_pm < pm:
                off1x = []
                off2x = []
                for j in range(x_num):
                    u2 = random.random()
                    if u2 < 0.5:
                        delta = float((2 * u2) ** (1 / yita2 + 1) - 1)
                    else:
                        delta = float(1 - (2 * (1 - u2)) ** (1 / (yita2 + 1)))
                    off1j = off1.x[j] + delta
                    off2j = off2.x[j] + delta
                    # 保证在定义域内
                    if off1j > x_max[0, j]:
                        off1j = x_max[0, j]
                    elif off1j < x_min[0, j]:
                        off1j = x_min[0, j]
                    if off2j > x_max[0, j]:
                        off2j = x_max[0, j]
                    elif off2j < x_min[0, j]:
                        off2j = x_min[0, j]
                    off1x.append(off1j)
                    off2x.append(off2j)
                off1 = Individual(off1x)
                off2 = Individual(off2x)

            population_offspring.append(off1)
            population_offspring.append(off2)
        return population_offspring

    # 精英机制
    def elitism(self, population_combine):
        assert len(population_combine) % 2 == 0
        N = int(len(population_combine) / 2)
        population_combine.sort()
        self.population = population_combine[0:N]
        return self.population


# 种群中的每个个体
class Individual():
    def __init__(self, x):
        self.x = x
        self.f = None
        self.crowd = 0
        self.pareto_rank = 0  # pareto等级 (从1开始)
        self.x_num = len(x)
        self.dominate_set = []  # 所支配的个体的集合，存放的是下标
        self.dominated_num = 0  # 被支配的个体的总数
        self.updateF()

    # 更新函数值
    def updateF(self):
        x_num = self.x_num
        x = self.x
        if fun_name == 'ZDT1':
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):  # 不加第一个值
                sum1 += x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** 0.5)
            self.f = [f1, f2]
        elif fun_name == 'ZDT2':
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(self.x_num - 1):
                sum1 += x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]
        elif fun_name == 'ZDT3':
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 += x[i + 1]
                g = float(1 + 9 * (sum1 / (x_num - 1)))
                f2 = g * (1 - (f1 / g) ** 0.5 - (f1 / g) * math.sin(10 * math.pi * f1))
                self.f = [f1, f2]
        elif fun_name == 'ZDT4':
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 += x[i + 1] ** 2 - 10 * math.cos(4 * math.pi * x[i + 1])
            g = float(1 + 9 * 10 + sum1)
            f2 = g * (1 - (f1 / g) ** 0.5)
            self.f = [f1, f2]
        elif fun_name == 'ZDT6':
            f1 = float(1 - math.exp(-4 * x[0]) * (math.sin(6 * math.pi * x[0])) ** 6)
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 += x[i + 1]
            g = float(1 + 9 * ((sum1 / (x_num - 1)) ** 0.25))
            f2 = g * (1 - (f1 / g) ** 2)
            self.f = [f1, f2]
        elif fun_name == 'DTLZ1':
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 += (x[i + 2] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i + 2] - 0.5))
                g = float(100 * (x_num - 2) + 100 * sum1)
                f1 = float((1 + g) * x[0] * x[1])
                f2 = float((1 + g) * x[0] * (1 - x[1]))
                f3 = float((1 + g) * (1 - x[0]))
                self.f = [f1, f2, f3]
        elif fun_name == 'DTLZ2':
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 += x[i + 2] ** 2
            g = float(sum1)
            f1 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.cos(0.5 * math.pi * x[1]))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.sin(0.5 * math.pi * x[1]))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * x[0]))
            self.f = [f1, f2, f3]

    # 比较方法,前提是pareto_rank 和crowd都赋值好
    def betterThan(self, indi2):
        assert isinstance(indi2, Individual)
        if self.pareto_rank < indi2.pareto_rank:
            return True
        elif self.pareto_rank > indi2.pareto_rank:
            return False
        else:
            return self.crowd >= indi2.crowd

    # 比较是否支配，使用f列表
    def dominate(self, indi2):
        less = greater = equal = 0
        f_num = len(self.f)
        for i in range(f_num):
            if self.f[i] > indi2.f[i]:
                greater += 1
            elif self.f[i] == indi2.f[i]:
                equal += 1
            else:
                less += 1
        if greater == 0 and equal != f_num:
            return True
        elif less == 0 and equal != f_num:
            return False
        return None

    # 重载小于号, 小于代表更优
    def __lt__(self, other):
        return self.betterThan(other)

    # 重载小于等于号
    def __le__(self, other):
        return self.betterThan(other)


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
        if p1[i] > p2.f[i]:
            greater += 1
        elif p1[i] == p2.f[i]:
            equal += 1
        else:
            less += 1
    if less == 0 and equal != f_num:
        return False
    elif greater == 0 and equal != f_num:
        return True
    return None


# 画出过程图
def draw_process_diagram(population, fun_name, f_num):
    fp = fun_name + ".txt"
    data = np.loadtxt(fp)
    if f_num == 2:
        x = []  # 注意不要用连等
        y = []
        for i in range(len(population)):
            x.append(population[i].f[0])
            y.append(population[i].f[1])
        plt.scatter(data[:, 0], data[:, 1], marker='o', color='green', s=40)  # best_solution 参考图
        plt.scatter(x, y, marker='o', color='red', s=40)
        plt.xlabel('f1' + str(iter))
        plt.ylabel('f2')
        plt.show()
    elif f_num == 3:
        x = []
        y = []
        z = []
        for i in range(len(population)):
            x.append(population[i].f[0])
            y.append(population[i].f[1])
            z.append(population[i].f[2])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='g')  # best_solution 参考图
        ax.scatter(x, y, z, c='r')
        plt.show()


def run_NSGA(fun_name, N, max_iter, pc, yita1, yita2, process_diagram=False):
    '''
    :param fun_name: 函数名
    :param N: 种群数量
    :param max_iter: 最带迭代次数
    :param pc: 交叉概率 （变异概率pm=1/x_num)
    :param yita1: 模拟二进制交叉参数
    :param yita2: 多项式变异参数
    :param process_diagram: 是否画出来过程图
    :return:
    '''
    start = time.time()
    f_num, x_num, x_min, x_max, best_solution = chooseFun(fun_name)
    pm = 1.0 / x_num
    # 构造NSGA2类，初始化种群
    GA = NSGA2(fun_name, x_num, N, f_num, max_iter, pc, pm, yita1, yita2, x_min, x_max)
    GA.init_population()

    # 加上非支配排序信息
    rank_indi_dict = GA.non_dominated_sort(GA.population)
    # 加上拥挤度信息,并赋值回去
    GA.population = GA.crowding_distance_sort(rank_indi_dict)

    iter = 1
    while iter <= max_iter:
        if iter % 1000 == 0:
            if process_diagram:
                draw_process_diagram(GA.population, fun_name, f_num)
            print(str(iter), " iterations completed")
        # 交叉和变异
        population_parent = GA.tournament_selection2(GA.population)
        population_offspring = GA.cross_mutation(population_parent)
        population_combine = population_parent + population_offspring
        assert len(population_combine) == 2 * GA.N

        # 此时x已经变化，种群信息也变化了，更新个体的F值等(修改后，子代是新对象，这里不用清空了)
        # for indi in population_combine:
        #     indi.crowd = 0
        #     indi.dominated_num = 0
        #     indi.dominate_set = []
        #     indi.pareto_rank = 0
        #     indi.updateF()

        # population_combine = sorted(population_combine, key=lambda Individual: Individual.f[0])

        # 加上非支配排序信息、拥挤度信息
        rank_indi_dict = GA.non_dominated_sort(population_combine)
        population_combine = GA.crowding_distance_sort(rank_indi_dict)
        # 精英选择，并赋值回GA对象
        GA.elitism(population_combine)
        iter += 1
    end = time.time()
    print("循环时间:%2f秒" % (end - start))
    draw_process_diagram(GA.population, fun_name, f_num)

    # coverage (C-metric) 评价
    number = 0
    for i in range(N):
        dominated_num = 0
        for j in range(len(best_solution)):
            if dominate(best_solution[j], GA.population[i]):
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
            for k in range(f_num):
                dd += float((best_solution[i][k] - GA.population[j].f[k]) ** 2)
            temp.append(math.sqrt(dd))
        min_d += np.min(temp)
    D_AP = float(min_d / N)
    print("D-metric: %2f" % D_AP)


if __name__ == "__main__":
    N = 150
    fun_list = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6', 'DTLZ1', 'DTLZ2']
    max_iter = 200 * 100
    pc = 0.8
    yita1 = 2
    yita2 = 5

    # fun_name = fun_list[6]
    # run_NSGA(fun_name, N, max_iter, pc, yita1, yita2, process_diagram=True)

    # run_NSGA(fun_name, N, max_iter, pc, yita1, yita2)

    for fun_name in fun_list:
        run_NSGA(fun_name, N, max_iter, pc, yita1, yita2)
