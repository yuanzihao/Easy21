import numpy as np
from utils.game import State, Action

def compute_MSE(Q, Q_star):
    return np.sum((Q[1:,1:,:] - Q_star)**2)

def feature_vector_pos(s:State, a:Action, dealer_cubio, player_cubio):
    dealer_S = dealer_cubio
    player_S = player_cubio
    x = np.zeros(36)
    for i, ds in enumerate(dealer_S):
        if s.dealer_sum in ds:
            for j, ps in enumerate(player_S):
                if s.agent_sum in ps:
                    if a == Action.STICK: x[i*12 + j] += 1
                    else: x[i*12 + 6 + j] += 1
                else: continue
        else: continue
    return x

def MapW2Q(w,Q,dealer_cubio, player_cubio):
    dealer_sum = np.arange(1,11)
    agent_sum = np.arange(1,22)
    for x in dealer_sum:
        for y in agent_sum:
            Q[x, y, 0] = np.dot(feature_vector_pos(State(x, y), Action.STICK, dealer_cubio=dealer_cubio, player_cubio=player_cubio), w)
            Q[x, y, 1] = np.dot(feature_vector_pos(State(x, y), Action.HIT, dealer_cubio=dealer_cubio, player_cubio=player_cubio), w)
    return Q