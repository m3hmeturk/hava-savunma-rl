"""
PPO Egitimi
===========

Ozel hedef onceliklendirme ortaminda PPO ajanini egitir.

Uretilenler:
  results/ppo_model.zip      -> egitilmis model (evaluate.py bunu otomatik yukler)
  results/learning_curve.png -> ogrenme egrisi (Bulgular bolumu icin)
  logs/                      -> TensorBoard kayitlari + monitor.csv

TensorBoard ile canli izlemek icin (ayri terminalde):
  tensorboard --logdir logs
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

sys.path.insert(0, "src/env")
from target_env import TargetPrioritizationEnv

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12

TOTAL_TIMESTEPS = 300_000      # egitim adim sayisi
LOG_DIR = "logs"
RESULTS_DIR = "results"
AHP_REFERENCE = 3.6            # ogrenme egrisinde referans cizgi


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
    ax.plot(x, y, alpha=0.20, color="#4285f4", label="Bölüm ödülü (ham)")
    ax.plot(x_smooth, y_smooth, color="#1a73e8", linewidth=2,
            label=f"Hareketli ortalama ({window})")
    ax.axhline(AHP_REFERENCE, color="#34a853", linestyle="--", linewidth=1.5,
               label=f"AHP referansı (~{AHP_REFERENCE})")
    ax.set_xlabel("Eğitim Adımı (timestep)")
    ax.set_ylabel("Bölüm Ödülü")
    ax.set_title("PPO Öğrenme Eğrisi")
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
    # ent_coef=0.01: kesif tesviki -> ajanin pasiflik lokal optimumuna
    # saplanmasini onler (atesleme stratejisini kesfetmesi icin gerekli)
    model = PPO("MlpPolicy", env, verbose=1, seed=0, ent_coef=0.01,
                tensorboard_log=LOG_DIR)

    print(f"Egitim basliyor: {TOTAL_TIMESTEPS} adim...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = os.path.join(RESULTS_DIR, "ppo_model")
    model.save(model_path)
    print(f"Model kaydedildi: {model_path}.zip")

    plot_learning_curve(LOG_DIR, os.path.join(RESULTS_DIR, "learning_curve.png"))
    print("Egitim tamamlandi.")


if __name__ == "__main__":
    main()
