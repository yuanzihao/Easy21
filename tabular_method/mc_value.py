import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.game import *
from utils.plot import *
from tqdm import tqdm
import copy

class Nums_Data:
    def __init__(self, dealer_max_nums, agent_max_nums, episodes):
        self.dealer_max_nums = dealer_max_nums
        self.agent_max_nums = agent_max_nums
        self.episodes = episodes

def montecarlo_vfn(data:Nums_Data):
    dealer_max_nums = data.dealer_max_nums
    agent_max_nums = data.agent_max_nums
    episodes = data.episodes
    game = Game()

    V = np.zeros((dealer_max_nums+1, agent_max_nums+1))
    N = np.zeros((dealer_max_nums+1, agent_max_nums+1))
    G_s = np.zeros((dealer_max_nums+1, agent_max_nums+1))
    player = Agent()
    for e in tqdm(range(episodes),desc="Monte Carlo"):
        step = []
        # if e % 50000 == 0: print(e)
        terminated = False
        s:State = State(game.initial_card(), game.initial_card(), False)
        a:Action = Action.HIT

        while not terminated:
            s_prime = copy.copy(s)
            a_prime:Action = copy.copy(a)

            s_prime, r, terminated = game.step(s, a)
            a_prime = player.policy(Q_input=None, s=s_prime)
            N[s.dealer_sum, s.agent_sum] += 1
            step.append((s, a, r))
            
            s = s_prime
            a = a_prime
        j = 0
        for temp_s, _, _ in step:
            G_t = sum([x[-1] for x in step[j:]])
            G_s[temp_s.dealer_sum, temp_s.agent_sum] += G_t
            V[temp_s.dealer_sum, temp_s.agent_sum] = G_s[temp_s.dealer_sum, temp_s.agent_sum] / N[temp_s.dealer_sum, temp_s.agent_sum]
            j += 1

    return V[1:, 1:]

if __name__ == '__main__':
    data = Nums_Data(dealer_max_nums=10, agent_max_nums=21, episodes=10)
    v = montecarlo_vfn(data)
    plot_v(data.dealer_max_nums, data.agent_max_nums, v)
    
    