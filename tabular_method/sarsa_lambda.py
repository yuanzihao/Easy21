import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.game import Action, State, Params, Game, Agent, Nums_Data
from utils.plot import plot_mse, plot_mse_group, plot_v
from utils.compute_fn import compute_MSE
from tqdm import tqdm
import numpy as np
import copy


def sarsa(data: Nums_Data, params:Params, q_star=None):
    dealer_max_nums = data.dealer_max_nums
    agent_max_nums = data.agent_max_nums
    episodes = data.episodes
    
    lambda_value = params.lambda_value
    gamma_value = params.gamma_value
    alpha_value = params.alpha_value

    game = Game()

    Q = np.zeros((dealer_max_nums+1, agent_max_nums+1, 2))
    player = Agent()
    mse = []
    for e in tqdm(range(episodes),desc="Sarsa Lambda"):
        terminated = False
        E = np.zeros((dealer_max_nums+1, agent_max_nums+1, 2))
        s:State = State(game.initial_card(), game.initial_card(), False)
        a:Action = Action.HIT

        while not terminated:
            s_prime = copy.copy(s)
            a_prime:Action = copy.copy(a)
            s_prime, r, terminated = game.step(s, a)
            if not terminated:
                a_prime = player.policy(Q_input=Q, s=s_prime, eps=0.05)
                TD_Error = r + gamma_value * Q[s_prime.dealer_sum, s_prime.agent_sum, a_prime.value] - Q[s.dealer_sum, s.agent_sum, a.value]
            else: TD_Error = r - Q[s.dealer_sum, s.agent_sum, a.value]
            E[s.dealer_sum, s.agent_sum, a.value] += 1
            Q = Q + alpha_value * TD_Error * E
            E = gamma_value * lambda_value * E
            
            s = s_prime
            a = a_prime
        
        if e % 1000 == 0:
            mse.append(compute_MSE(Q, q_star))
    return Q[1:, 1:, :], mse

if __name__ == '__main__':
    # params = Params(lambda_value=1, gamma_value=1, alpha_value=0.03)
    data = Nums_Data(dealer_max_nums=10, agent_max_nums=21, episodes=10)
    q_star = np.load("tabular_method\q_star.npy")
    _lambda = np.arange(0, 1.0, 0.1)
    t_mse = []
    for l in _lambda:
        print(f"lambda is {l}")
        params = Params(lambda_value=l, gamma_value=1, alpha_value=0.03)
        q, mse = sarsa(data, params, q_star)
        t_mse.append(mse)
    v = np.max(q,axis=-1)
    plot_mse_group(t_mse)
    # plot_v(data.dealer_max_nums, data.agent_max_nums, v)
    
    
    