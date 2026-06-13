import sys
sys.path.insert(0, "src/env")
sys.path.insert(0, "src/agents")
import numpy as np
from target_env import TargetPrioritizationEnv
from baselines import RandomAgent, AHPAgent

env = TargetPrioritizationEnv()
ahp = AHPAgent()
print("AHP agirliklari [Tehdit, Yakinlik, Hiz]:", np.round(ahp.weights, 4), "| CR:", round(ahp.cr, 4))

def run(agent, n=50):
    rew = []; stats = {"neutralized": 0, "leaked": 0, "friendly_fire": 0}
    for ep in range(n):
        obs, info = env.reset(seed=5000 + ep); done = False; R = 0; last = info
        while not done:
            obs, r, term, trunc, last = env.step(agent.act(obs)); R += r; done = term or trunc
        rew.append(R)
        for k in stats: stats[k] += last[k]
    return np.mean(rew), stats

for name, ag in [("Rastgele", RandomAgent(env.action_space)), ("AHP", ahp)]:
    m, s = run(ag)
    print(f"{name:<9} ort.odul {m:7.2f} | {s}")