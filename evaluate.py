"""
Degerlendirme ve Sonuc Uretimi
==============================

Tum ajanlari (Rastgele, AHP, ve varsa egitilmis PPO) AYNI test
senaryolarinda kosturur, metrikleri hesaplar ve rapora hazir ciktilar
uretir:

  results/comparison_table.csv      -> Bulgular bolumu icin tablo
  results/reward_comparison.png     -> ortalama odul karsilastirmasi
  results/metrics_comparison.png    -> etkisizlestirme/sizinti/dost-atesi

PPO modeli (results/ppo_model.zip) yoksa sadece baseline'lar cizilir;
Faz 3'te model egitilince ayni betik PPO'yu otomatik ekler.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ekransiz ortamda da calissin
import matplotlib.pyplot as plt

sys.path.insert(0, "src/env")
sys.path.insert(0, "src/agents")
from target_env import TargetPrioritizationEnv
from baselines import RandomAgent, AHPAgent

# Rapor ile ayni yazi tipi (Times New Roman); yoksa serif'e duser
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

N_EPISODES = 200          # degerlendirme bolumu sayisi
RESULTS_DIR = "results"
PPO_PATH = os.path.join(RESULTS_DIR, "ppo_model.zip")


# ----------------------------------------------------------------------
# PPO modelini saran kucuk yardimci (act arayuzu baseline'larla ayni olsun)
# ----------------------------------------------------------------------
class PPOWrapper:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_agents(env):
    agents = {
        "Rastgele": RandomAgent(env.action_space),
        "AHP": AHPAgent(),
    }
    if os.path.exists(PPO_PATH):
        from stable_baselines3 import PPO
        agents["PPO"] = PPOWrapper(PPO.load(PPO_PATH))
        print(f"PPO modeli yuklendi: {PPO_PATH}")
    else:
        print("PPO modeli bulunamadi -> sadece baseline'lar degerlendiriliyor.")
    return agents


def evaluate(agent, env, n=N_EPISODES):
    """Sabit seed'lerle n bolum kosturur; bolum odullerini ve toplam
    istatistikleri dondurur (tum ajanlar ayni senaryolari gorur)."""
    rewards = []
    stats = {"neutralized": 0, "leaked": 0, "friendly_fire": 0}
    for ep in range(n):
        obs, info = env.reset(seed=10_000 + ep)
        done = False
        ep_reward = 0.0
        last = info
        while not done:
            obs, r, term, trunc, last = env.step(agent.act(obs))
            ep_reward += r
            done = term or trunc
        rewards.append(ep_reward)
        for k in stats:
            stats[k] += last[k]
    rewards = np.array(rewards)
    return {
        "mean_reward": rewards.mean(),
        "std_reward": rewards.std(),
        **stats,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = TargetPrioritizationEnv()
    agents = load_agents(env)

    # --- Degerlendir ---
    rows = []
    for name, agent in agents.items():
        res = evaluate(agent, env)
        res["agent"] = name
        rows.append(res)
        print(f"{name:<10} ort.odul {res['mean_reward']:7.2f} +/- {res['std_reward']:5.2f}"
              f" | etkisiz:{res['neutralized']:4d} sizinti:{res['leaked']:3d}"
              f" dost-atesi:{res['friendly_fire']:4d}")

    df = pd.DataFrame(rows)[
        ["agent", "mean_reward", "std_reward", "neutralized", "leaked", "friendly_fire"]
    ]
    df.columns = ["Ajan", "Ort.Ödül", "Std", "Etkisizleştirme", "Sızıntı", "Dost Ateşi"]

    # --- CSV tablo ---
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nTablo kaydedildi: {csv_path}")

    names = df["Ajan"].tolist()
    colors = ["#9aa0a6", "#4285f4", "#34a853"][: len(names)]

    # --- Grafik 1: ortalama odul ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, df["Ort.Ödül"], yerr=df["Std"], capsize=6, color=colors,
           edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Ortalama Bölüm Ödülü")
    ax.set_title(f"Yöntemlere Göre Ortalama Ödül (n={N_EPISODES})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(RESULTS_DIR, "reward_comparison.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    print(f"Grafik kaydedildi: {p1}")

    # --- Grafik 2: metrik karsilastirmasi (gruplu cubuk) ---
    metrics = ["Etkisizleştirme", "Sızıntı", "Dost Ateşi"]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(names):
        vals = df[df["Ajan"] == name][metrics].values.flatten()
        ax.bar(x + i * width, vals, width, label=name, color=colors[i],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel(f"Toplam Sayım ({N_EPISODES} bölüm)")
    ax.set_title("Yöntemlerin Davranış Metrikleri")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    print(f"Grafik kaydedildi: {p2}")


if __name__ == "__main__":
    main()
