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
DQN_PATH = os.path.join(RESULTS_DIR, "dqn_model.zip")


# ----------------------------------------------------------------------
# SB3 modelini saran kucuk yardimci (act arayuzu baseline'larla ayni olsun)
# ----------------------------------------------------------------------
class SB3Wrapper:
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
        agents["PPO"] = SB3Wrapper(PPO.load(PPO_PATH))
        print(f"PPO modeli yuklendi: {PPO_PATH}")
    else:
        print("PPO modeli bulunamadi.")
    if os.path.exists(DQN_PATH):
        from stable_baselines3 import DQN
        agents["DQN"] = SB3Wrapper(DQN.load(DQN_PATH))
        print(f"DQN modeli yuklendi: {DQN_PATH}")
    else:
        print("DQN modeli bulunamadi.")
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
    palette = {"Rastgele": "#9aa0a6", "AHP": "#4285f4", "PPO": "#34a853", "DQN": "#c2410c"}
    colors = [palette.get(n, "#999999") for n in names]

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

    # --- Grafik 3: birlesik ogrenme egrisi (PPO vs DQN) ---
    plot_combined_learning_curves()


def _load_curve(candidate_dirs):
    """Verilen aday klasorlerden ilk bulunan monitor logunu (x, y) olarak dondurur."""
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    for d in candidate_dirs:
        try:
            if os.path.exists(d) and any(f.endswith(".monitor.csv") for f in os.listdir(d)):
                x, y = ts2xy(load_results(d), "timesteps")
                if len(y) > 0:
                    return x, y
        except Exception:
            continue
    return None, None


def _smooth(values, window):
    window = min(window, max(1, len(values) // 10))
    w = np.ones(window) / window
    return np.convolve(values, w, mode="valid"), window


def plot_combined_learning_curves():
    """PPO ve DQN ogrenme egrilerini ayni eksende cizer."""
    px, py = _load_curve([os.path.join("logs", "ppo"), "logs"])
    dx, dy = _load_curve([os.path.join("logs", "dqn")])
    if py is None and dy is None:
        print("Birlesik egri icin log bulunamadi (egitim loglari yok), atlandi.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if py is not None:
        ys, w = _smooth(py, 50)
        ax.plot(px[len(px) - len(ys):], ys, color="#1a73e8", linewidth=2,
                label=f"PPO (hareketli ort. {w})")
    if dy is not None:
        ys, w = _smooth(dy, 50)
        ax.plot(dx[len(dx) - len(ys):], ys, color="#c2410c", linewidth=2,
                label=f"DQN (hareketli ort. {w})")
    ax.axhline(3.6, color="#34a853", linestyle="--", linewidth=1.5,
               label="AHP referansı (~3.6)")
    ax.set_xlabel("Eğitim Adımı (timestep)")
    ax.set_ylabel("Bölüm Ödülü")
    ax.set_title("PPO ve DQN Öğrenme Eğrileri Karşılaştırması")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(RESULTS_DIR, "learning_curves_combined.png")
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    print(f"Grafik kaydedildi: {p3}")


if __name__ == "__main__":
    main()
