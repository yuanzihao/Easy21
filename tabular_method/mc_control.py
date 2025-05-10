import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.game import Action, State, Params, Game, Agent, Nums_Data
from utils.plot import plot_mse, plot_mse_group, plot_v
from tqdm import tqdm
import numpy as np
import copy

class Nums_Data:
    def __init__(self, dealer_max_nums, agent_max_nums, episodes):
        self.dealer_max_nums = dealer_max_nums
        self.agent_max_nums = agent_max_nums
        self.episodes = episodes

def montecarlo(data:Nums_Data):
    dealer_max_nums = data.dealer_max_nums
    agent_max_nums = data.agent_max_nums
    episodes = data.episodes
    game = Game()
    Q = np.zeros((dealer_max_nums+1, agent_max_nums+1, 2))
    N = np.ones((dealer_max_nums+1, agent_max_nums+1, 2))
    player = Agent()
    for e in tqdm(range(episodes),desc="Monte Carlo"):
        step = []
        terminated = False
        s:State = State(game.initial_card(), game.initial_card(), False)
        a:Action = Action.HIT
        
        while not terminated:
            s_prime = copy.copy(s)
            a_prime: Action = copy.copy(a)
            s_prime, r, terminated = game.step(s, a)
            if not terminated:
                a_prime = player.policy(Q_input=Q, s=s_prime, eps=1 / N[s.dealer_sum, s.agent_sum, a.value])
            step.append((s, a, r))
            s = s_prime
            a = a_prime
        j = 0
        for temp_s, temp_a, _ in step:
            N[temp_s.dealer_sum, temp_s.agent_sum, temp_a.value] += 1
            G_t = sum([x[-1] for _, x in enumerate(step[j:])])
            Q[temp_s.dealer_sum, temp_s.agent_sum, temp_a.value] += (G_t - Q[temp_s.dealer_sum, temp_s.agent_sum, temp_a.value]) / N[temp_s.dealer_sum, temp_s.agent_sum, temp_a.value]
            j += 1

    return Q[1:, 1:, :]

if __name__ == '__main__':
    data = Nums_Data(dealer_max_nums=10, agent_max_nums=21, episodes=10)
    q = montecarlo(data)
    v = np.max(q,axis=-1)
    # np.save("q_star.npy", q)
    plot_v(data.dealer_max_nums, data.agent_max_nums, v)
    
    