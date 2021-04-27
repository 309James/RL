import pygame
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from RL_2_YuanyangForMC import YuanYangEnv


class MC_RL:
    def __init__(self, yuanyang):
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions))) * 0.1
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        self.actions = yuanyang.actions
        self.yuanyang = yuanyang
        self.gamma = yuanyang.gamma

    def greedy_policy(self, qfun, state):
        amax = qfun[state, :].argmax()
        return self.actions[amax]

    def epsilon_greedy_policy(self, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        if np.random.uniform() < 1 - epsilon:
            return self.actions[amax]
        else:
            return self.actions[int(random.random() * len(self.actions))]

    def find_anum(self, a):
        for i in range(len(self.actions)):
            if a == self.actions[i]:
                return i

    def mc_learning_ei(self, num_iter):
        yuanyang = self.yuanyang
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions)))
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        for iter1 in range(num_iter):
            s_sample = []
            a_sample = []
            r_sample = []
            s = self.yuanyang.reset()
            a = self.actions[int(random.random() * len(self.actions))]
            done = False
            step_num = 0

            if self.mc_test() == 1:
                print('探索初始化第一次完成任务需要的迭代次数：', iter1)
                break
            # 采集数据s0-a1-s1-a2-s2...terminate state
            while done == False and step_num < 30:
                s_next, r, done = self.yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                # 往回走不合理，因此给予负数回报
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                a_sample.append(a_num)
                r_sample.append(r)
                step_num += 1
                s = s_next
                a = self.greedy_policy(self.qvalue, s)
            #         计算折扣累计回报
            # g(s_0)
            a = self.greedy_policy(self.qvalue, s)
            g = self.qvalue[s, self.find_anum(a)]
            for i in range(len(s_sample) - 1, -1, -1):
                g *= self.gamma
                g += r_sample[i]
            # g=G(s1,a)
            for i in range(len(s_sample)):
                self.n[s_sample[i], a_sample[i]] += 1.0
                self.qvalue[s_sample[i], a_sample[i]] = \
                    (self.qvalue[s_sample[i], a_sample[i]] * (self.n[s_sample[i],
                                                                     a_sample[i]] - 1) + g) / self.n[s_sample[i],
                                                                                                     a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
        return self.qvalue

    def mc_learning_on_policy(self,num_iter,epsilon):
        yuanyang = self.yuanyang
        self.qvalue = np.zeros((len(yuanyang.states), len(yuanyang.actions)))
        self.n = 0.001 * np.ones((len(yuanyang.states), len(yuanyang.actions)))
        for iter1 in range(num_iter):
            s_sample=[]
            r_sample=[]
            a_sample=[]
            s=0
            done=False
            step_num=0
            epsilon=epsilon*np.exp(-iter1/1000)
            while done == False and step_num < 30:
                a=self.epsilon_greedy_policy(self.qvalue, s, epsilon)
                s_next,r,done=yuanyang.transform(s,a)
                a_num=self.find_anum(a)
                # 往回走不合理，因此给予负数回报
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                a_sample.append(a_num)
                r_sample.append(r)
                step_num += 1
                s = s_next
            if self.mc_test() == 1:
                print('探索初始化第一次完成任务需要的迭代次数：', iter1)
                break
            a=self.epsilon_greedy_policy(self.qvalue,s,epsilon)
            g=self.qvalue[s,self.find_anum(a)]
            for i in range(len(s_sample)-1,-1,-1):
                g*=self.gamma
                g+=r_sample[i]

            for i in range(len(s_sample)):
                self.n[s_sample[i],a_sample[i]]+=1.0
                self.qvalue[s_sample[i],a_sample[i]]= \
                    (self.qvalue[s_sample[i], a_sample[i]] * (self.n[s_sample[i],
                                                                     a_sample[i]] - 1) + g) / self.n[s_sample[i],
                                                                                                     a_sample[i]]
                g -= r_sample[i]
                g /= self.gamma
        return self.qvalue

    def mc_test(self):
        s = 0
        s_sample = []
        done = False
        flag = 0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy(self.qvalue, s)
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 9:
            flag = 1
        return flag


if __name__ == '__main__':
    yuanyang = YuanYangEnv()
    brain = MC_RL(yuanyang)
    qvalue1 = brain.mc_learning_on_policy(num_iter=10000,epsilon=0.15)
    yuanyang.action_value = qvalue1
    flag = 1
    s = 0
    step_num = 0
    path = []
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy(qvalue1, s)
        print('%d -> %s' % (s, a), qvalue1[s, 0], qvalue1[s, 1], qvalue1[s, 2], qvalue1[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 30:
            flag = 0
        s = s_

    yuanyang.bird_male_position = yuanyang.state_to_position(s)
    path.append(s)
    yuanyang.render()
    while True:
        yuanyang.render()
