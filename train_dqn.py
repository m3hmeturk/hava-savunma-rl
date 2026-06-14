"""
DQN Egitimi
===========

Ozel hedef onceliklendirme ortaminda DQN (Deep Q-Network) ajanini egitir.
PPO ile ADIL karsilastirma icin ayni adim sayisi (300k) kullanilir.

PPO (politika-gradyan, on-policy) vs DQN (deger-tabanli, off-policy):
iki farkli RL paradigmasini ayni problemde karsilastirmak icin.

Uretilenler:
  results/dqn_model.zip          -> egitilmis model (evaluate.py otomatik yukler)
  results/learning_curve_dqn.png -> DQN ogrenme egrisi
  logs/dqn/                      -> TensorBoard + monitor.csv

DQN icin onemli ayar: exploration_fraction=0.5 ile epsilon-greedy kesif
suresi uzun tutulur; aksi halde ajan pasiflik lokal optimumuna saplanir.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

sys.path.insert(0, "src/env")
from target_env import TargetPrioritizationEnv

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

TOTAL_TIMESTEPS = 300_000      # PPO ile ayni (adil karsilastirma)
LOG_DIR = os.path.join("logs", "dqn")
RESULTS_DIR = "results"
AHP_REFERENCE = 3.6


def moving_average(values, window):
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_learning_curve(log_dir, out_path):
    x, y = ts2xy(load_results(log_dir), "timesteps")
    if len(y) == 0:
        print("Uyari: ogrenme egrisi icin veri bulunamadi.")
        return
    window = min(50, max(1, len(y) // 10))
    y_smooth = moving_average(y, window)
    x_smooth = x[len(x) - len(y_smooth):]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, alpha=0.20, color="#e07b39", label="Bölüm ödülü (ham)")
    ax.plot(x_smooth, y_smooth, color="#c2410c", linewidth=2,
            label=f"Hareketli ortalama ({window})")
    ax.axhline(AHP_REFERENCE, color="#34a853", linestyle="--", linewidth=1.5,
               label=f"AHP referansı (~{AHP_REFERENCE})")
    ax.set_xlabel("Eğitim Adımı (timestep)")
    ax.set_ylabel("Bölüm Ödülü")
    ax.set_title("DQN Öğrenme Eğrisi")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Öğrenme eğrisi kaydedildi: {out_path}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = Monitor(TargetPrioritizationEnv(seed=0), os.path.join(LOG_DIR, "monitor"))
    model = DQN(
        "MlpPolicy", env, verbose=1, seed=0,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        exploration_fraction=0.5,        # uzun kesif -> pasiflik tuzagindan kacin
        exploration_final_eps=0.05,
        tensorboard_log=LOG_DIR,
    )

    print(f"DQN egitimi basliyor: {TOTAL_TIMESTEPS} adim...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = os.path.join(RESULTS_DIR, "dqn_model")
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}.zip")

    plot_learning_curve(LOG_DIR, os.path.join(RESULTS_DIR, "learning_curve_dqn.png"))
    print("DQN egitimi tamamlandi.")


if __name__ == "__main__":
    main()
