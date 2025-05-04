import numpy as np
from enum import Enum
import copy


class Nums_Data:
    def __init__(self, dealer_max_nums, agent_max_nums, episodes, eps=0.05, step_size=0.01):
        self.dealer_max_nums = dealer_max_nums
        self.agent_max_nums = agent_max_nums
        self.episodes = episodes
        self.eps = eps
        self.step_size = step_size


class Action(Enum):
    STICK = 0
    HIT = 1

class State:
    def __init__(self, dealer_sum, agent_sum, is_terminal=False):
        self.dealer_sum = dealer_sum
        self.agent_sum = agent_sum
    
    def __str__(self):
        return f'{self.dealer_sum, self.agent_sum}'

class Params:
    def __init__(self, lambda_value, gamma_value, alpha_value):
        self.lambda_value = lambda_value
        self.gamma_value = gamma_value
        self.alpha_value = alpha_value

class Game:
    def __init__(self):
        self.color = np.array([-1, 1])
        self.num = np.arange(1, 11)
    
    def initial_card(self):
        return np.random.choice(self.num)

    def take_card(self):
        return np.random.choice(self.color, p=[1./3., 2./3.]) * np.random.choice(self.num)
    
    def step(self, s:State, a:Action):
        # 在这里实现牌桌的推进
        s_now = copy.copy(s)
        s_prime = copy.copy(s)
        if a == Action.HIT:
            # 取牌的情况
            s_prime.agent_sum += self.take_card()
            if s_prime.agent_sum > 21 or s_prime.agent_sum < 1:
                reward = -1
                terminated = True
            else:
                reward = 0
                terminated = False
        else:
            # 不取牌的情况
            terminated = True
            while s_now.dealer_sum < 17 and s_now.dealer_sum >= 1: 
                s_now.dealer_sum += self.take_card()
            if s_now.dealer_sum > 21 or s_now.dealer_sum < 1: 
                reward = 1
                return s_prime, reward, terminated
            return s_prime, np.sign(s_now.agent_sum - s_now.dealer_sum), terminated
        return s_prime, reward, terminated
    
class Agent:
    def __init__(self, action=Action.HIT):
        self.action = action
        self.policy_p = [0.5, 0.5]
    
    def epsilon_greedy(self, Q_input=None, s:State=None, eps=0.05):
        d_sum = s.dealer_sum
        a_sum = s.agent_sum
        pi_max = np.argmax(Q_input[d_sum,a_sum,:])
        if pi_max == 0: return [eps/2.+1.-eps, eps/2.]
        else: return [eps/2., eps/2.+1.-eps]
    def policy(self, Q_input=None, s:State=None, eps=0.05):
        if Q_input is None: return Action.STICK if s.agent_sum >= 17 else Action.HIT
        self.policy_p = self.epsilon_greedy(Q_input, s, eps)
        return np.random.choice([Action.STICK, Action.HIT], p=self.policy_p)
    
    def linear_policy(self, flag=0, eps=0.05):
        if flag == 0: return np.random.choice([Action.STICK, Action.HIT], p=[eps/2.+1.-eps, eps/2.])
        else: return np.random.choice([Action.STICK, Action.HIT], p=[eps/2., eps/2.+1.-eps])
    




