import sys
sys.path.insert(0, "src/env")
from target_env import TargetPrioritizationEnv
from stable_baselines3.common.env_checker import check_env

env = TargetPrioritizationEnv(seed=42)
check_env(env, warn=True)
print("check_env: GECTI — ortam SB3 ile uyumlu.")