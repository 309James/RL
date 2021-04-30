from RL_2_YuanyangForMC import *
from RL_2_YuanyangForMC import YuanYangEnv
import time


class LFA_RL:
    def __init__(self, yuanyang):
        self.gamma = yuanyang.gamma
        self.yuanyang = yuanyang
        self.theta_tr = 0.1 * np.zeros((400, 1))
        self.theta_fsr = 0.1 * np.zeros((80, 1))

    def find_anum(self, a):
        for i in range(len(self.yuanyang.actions)):
            if a == self.yuanyang.actions[i]:
                return i

    def feature_tr(self, s, a):
        phi_s_a = np.zeros((1, 400))
        phi_s_a[0, 100 * a + s] = 1
        return phi_s_a

    def greedy_policy_tr(self, state):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        return self.yuanyang.actions[amax]

    def epsilon_greedy_policy_tr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        for i in range(4):
            qfun[i] = np.dot(self.feature_tr(state, i), self.theta_tr)
        amax = qfun.argmax()
        if np.random.uniform() < 1 - epsilon:
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]

    def greedy_test_tr(self):
        s = 0
        s_sample = []
        done = False
        flag=0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy_tr(s)
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 9:
            flag = 1
        if s == 9 and step_num < 21:
            flag = 2
        return flag

    def q_learning_lfa_tr(self, num_iter, alpha, epsilon):
        yuanyang = self.yuanyang
        iter_num = []
        self.theta_tr = np.zeros((400, 1)) * 0.1
        for iter in range(num_iter):
            s = 0
            s_sample = []
            flag = self.greedy_test_tr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('sarsa第一次完成任务需要迭代的次数为：', iter_num[0])
            if flag == 2:
                print('sarsa第一次实现最短路径需要迭代的次数为：', iter)
                break
            a = self.epsilon_greedy_policy_tr(s, epsilon)
            t = False
            count = 0
            while (False==t) and count < 30:
                s_next, r, t = yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    a1 = self.greedy_policy_tr(s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * np.dot(self.feature_tr(s_next, a1_num), self.theta_tr)
                self.theta_tr = self.theta_tr + alpha * (
                            q_target - np.dot(self.feature_tr(s, a_num), self.theta_tr))[0, 0] * np.transpose(self.feature_tr(s, a_num))
                s = s_next
                a = self.epsilon_greedy_policy_tr(s, epsilon)
                count += 1
        return self.theta_tr

    def feature_fsr(self,s,a):
        phi_s_a=np.zeros((1,80))
        y=int(s/10)
        x=s-10*y
        phi_s_a[0,20*a+x] = 1
        phi_s_a[0,20*a+10+y]=1
        return phi_s_a

    def greedy_policy_fsr(self,state):
        qfun=0.1*np.array([0,0,0,0])
        for i in range(4):
            qfun[i]=np.dot(self.feature_fsr(state,i),self.theta_fsr)
        amax=qfun.argmax()
        return self.yuanyang.actions[amax]

    def epsilon_greedy_policy_fsr(self, state, epsilon):
        qfun = np.array([0, 0, 0, 0]) * 0.1
        for i in range(4):
            qfun[i] = np.dot(self.feature_fsr(state, i), self.theta_fsr)
        amax = qfun.argmax()
        if np.random.uniform() < 1 - epsilon:
            return self.yuanyang.actions[amax]
        else:
            return self.yuanyang.actions[int(random.random() * len(self.yuanyang.actions))]

    def greedy_test_fsr(self):
        s = 0
        s_sample = []
        done = False
        flag=0
        step_num = 0
        while False == done and step_num < 30:
            a = self.greedy_policy_fsr(s)
            s_next, r, done = self.yuanyang.transform(s, a)
            s_sample.append(s)
            s = s_next
            step_num += 1
        if s == 9:
            flag = 1
        if s == 9 and step_num < 21:
            flag = 2
        return flag

    def q_learning_lfa_fsr(self,num_iter,alpha,epsilon):
        iter_num=[]
        self.theta_fsr = np.zeros((80, 1)) * 0.1
        for iter in range(num_iter):
            s = 0
            s_sample = []
            flag = self.greedy_test_fsr()
            if flag == 1:
                iter_num.append(iter)
                if len(iter_num) < 2:
                    print('sarsa第一次完成任务需要迭代的次数为：', iter_num[0])
            if flag == 2:
                print('sarsa第一次实现最短路径需要迭代的次数为：', iter)
                break
            a = self.epsilon_greedy_policy_fsr(s, epsilon)
            t = False
            count = 0
            while (False==t) and count < 30:
                s_next, r, t = yuanyang.transform(s, a)
                a_num = self.find_anum(a)
                if s_next in s_sample:
                    r = -2
                s_sample.append(s)
                if t == True:
                    q_target = r
                else:
                    a1 = self.greedy_policy_fsr(s_next)
                    a1_num = self.find_anum(a1)
                    q_target = r + self.gamma * np.dot(self.feature_fsr(s_next, a1_num), self.theta_fsr)
                self.theta_fsr = self.theta_fsr + alpha * (
                        q_target - np.dot(self.feature_fsr(s, a_num), self.theta_fsr))[0, 0] * np.transpose(self.feature_fsr(s, a_num))
                s = s_next
                a = self.epsilon_greedy_policy_fsr(s, epsilon)
                count += 1
        return self.theta_fsr


if __name__ == '__main__':
    yuanyang = YuanYangEnv()
    brain = LFA_RL(yuanyang)
    brain.q_learning_lfa_fsr(num_iter=10000, alpha=0.1, epsilon=0.1)
    qvalue1 = np.zeros((100, 4))
    for i in range(400):
        y = int(i / 100)
        x = i - 100 * y
        qvalue1[x, y] = np.dot(brain.feature_fsr(x, y), brain.theta_fsr)
    yuanyang.action_value = qvalue1
    flag = 1
    s = 0
    step_num = 0
    path = []
    while flag:
        path.append(s)
        yuanyang.path = path
        a = brain.greedy_policy_fsr(s)
        print('%d -> %s' % (s, a), qvalue1[s, 0], qvalue1[s, 1], qvalue1[s, 2], qvalue1[s, 3])
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.25)
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
