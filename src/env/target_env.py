"""
Hava Savunma Sistemi icin Hedef Onceliklendirme Ortami
=======================================================

2 boyutlu, basit kinematik bir simulasyon. Savunma bolgesi merkezde (0,0).
Hedefler karsidan, sabit hizla, yavasca ve slalom (yanal sinus salinimi)
yaparak yaklasir. Taret her adimda hangi hedefe yonelecegine/atesleyecegine
karar verir. Amac: yuksek tehditli dusman hedefleri savunma cizgisini
gecmeden etkisizlestirmek; dostlara ates etmemek.

MDP ozeti:
  - State : 47 boyut  (5 hedef slotu x 9 ozellik + 2 taret durumu)
  - Action: Discrete(6) (0-4 hedef sec, 5 bekle)
  - Reward: dusman etkisizlestirme (+), dosta ates (--), sizinti (-),
            zaman cezasi (-), taret donus cezasi (-), gecersiz secim (-)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TargetPrioritizationEnv(gym.Env):
    """Hava savunma hedef onceliklendirme ortami."""

    metadata = {"render_modes": []}

    # ----- Sabit parametreler -----
    MAX_TARGETS = 5          # ayni anda en fazla hedef slotu
    MAX_STEPS = 300          # bolum basina maksimum adim
    DEFENSE_RADIUS = 10.0    # bu cizgiyi gecen dusman = sizinti (leak)
    SPAWN_DISTANCE = 100.0   # hedeflerin baslangic mesafesi
    TRACK_RANGE = 80.0       # bu mesafenin altinda otonom takip hazir
    FIRE_RANGE = 60.0        # bu mesafenin altinda otonom vurus hazir
    MIN_SPEED = 0.30         # yavas, sabit hiz alt siniri (birim/adim)
    MAX_SPEED = 0.60         # ust sinir
    BEARING_FAN = math.radians(60.0)  # +/- 60 derece on yelpaze
    SLALOM_AMP = math.radians(12.0)   # slalom yanal salinim genligi
    SLALOM_FREQ = 0.10                # slalom frekansi
    COOLDOWN = 3             # ates sonrasi bekleme (adim)
    MAX_THREAT = 3           # tehdit seviyesi 1..3
    ENEMY_PROB = 0.70        # uretilen hedeflerin dusman olma olasiligi

    # ----- Odul katsayilari -----
    R_NEUTRALIZE = 1.0       # x tehdit seviyesi
    R_FRIENDLY_FIRE = -2.0
    R_LEAK = -1.0
    R_TIME = -0.01
    R_SLEW = -0.05           # x normalize aci farki
    R_INVALID = -0.02

    def __init__(self, render_mode=None, seed=None):
        super().__init__()
        self.render_mode = render_mode

        # Action: 0..4 hedef slotu sec, 5 = bekle
        self.action_space = spaces.Discrete(self.MAX_TARGETS + 1)

        # Observation: 47 boyut, hepsi [-1, 1] araliginda normalize
        obs_dim = self.MAX_TARGETS * 9 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    # Hedef uretimi
    # ----------------------------------------------------------------
    def _spawn_targets(self):
        """Bolum basinda 3-5 hedef uretir."""
        n = int(self._rng.integers(3, self.MAX_TARGETS + 1))  # 3..5
        targets = []
        for _ in range(n):
            base_bearing = self._rng.uniform(-self.BEARING_FAN, self.BEARING_FAN)
            targets.append({
                "active": True,
                "distance": self.SPAWN_DISTANCE * self._rng.uniform(0.9, 1.0),
                "speed": self._rng.uniform(self.MIN_SPEED, self.MAX_SPEED),
                "base_bearing": base_bearing,
                "bearing": base_bearing,
                "phase": self._rng.uniform(0.0, 2 * math.pi),
                "threat": int(self._rng.integers(1, self.MAX_THREAT + 1)),  # 1..3
                "is_enemy": bool(self._rng.random() < self.ENEMY_PROB),
            })
        # Sabit 5 slota yerlestir; bos slotlar None
        slots = [None] * self.MAX_TARGETS
        for i, t in enumerate(targets):
            slots[i] = t
        return slots

    # ----------------------------------------------------------------
    # Yardimcilar
    # ----------------------------------------------------------------
    @staticmethod
    def _track_ready(t):
        return t is not None and t["active"] and t["distance"] < TargetPrioritizationEnv.TRACK_RANGE

    @staticmethod
    def _fire_ready(t):
        return (t is not None and t["active"]
                and t["distance"] < TargetPrioritizationEnv.FIRE_RANGE
                and t["distance"] < TargetPrioritizationEnv.TRACK_RANGE)

    def _time_to_reach(self, t):
        if t is None or not t["active"]:
            return 0.0
        return max(0.0, (t["distance"] - self.DEFENSE_RADIUS) / t["speed"])

    def _get_obs(self):
        obs = []
        for t in self.slots:
            if t is None or not t["active"]:
                # bos / pasif slot -> sifir vektor
                obs.extend([0.0] * 9)
                continue
            ttr = self._time_to_reach(t)
            obs.extend([
                1.0,                                                  # 1 aktif
                np.clip(t["distance"] / self.SPAWN_DISTANCE, 0, 1),   # 2 mesafe
                np.clip(t["speed"] / self.MAX_SPEED, 0, 1),           # 3 hiz
                np.clip(t["bearing"] / (math.pi / 2), -1, 1),         # 4 kerteriz
                t["threat"] / self.MAX_THREAT,                        # 5 tehdit
                1.0 if t["is_enemy"] else 0.0,                        # 6 dusman mi
                1.0 if self._track_ready(t) else 0.0,                 # 7 takip hazir
                1.0 if self._fire_ready(t) else 0.0,                  # 8 vurus hazir
                np.clip(ttr / self.MAX_STEPS, 0, 1),                  # 9 erisim suresi
            ])
        # Taret durumu
        obs.append(np.clip(self.aim_angle / (math.pi / 2), -1, 1))    # nisan acisi
        obs.append(self.cooldown / self.COOLDOWN)                     # cooldown
        return np.array(obs, dtype=np.float32)

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.slots = self._spawn_targets()
        self.aim_angle = 0.0
        self.cooldown = 0
        self.steps = 0
        info = {"neutralized": 0, "leaked": 0, "friendly_fire": 0}
        self._stats = dict(info)
        return self._get_obs(), info

    def step(self, action):
        action = int(action)
        reward = 0.0
        self.steps += 1

        # --- 1) Aksiyonu uygula ---
        if action < self.MAX_TARGETS:
            t = self.slots[action]
            if t is None or not t["active"] or not self._fire_ready(t):
                # bos slot ya da vurusa hazir olmayan hedef -> gecersiz
                reward += self.R_INVALID
            else:
                # taret donus (slew) cezasi
                target_bearing = t["bearing"]
                slew = abs(target_bearing - self.aim_angle) / math.pi
                reward += self.R_SLEW * slew
                self.aim_angle = target_bearing

                if self.cooldown == 0:
                    if t["is_enemy"]:
                        reward += self.R_NEUTRALIZE * t["threat"]
                        t["active"] = False
                        self._stats["neutralized"] += 1
                    else:
                        reward += self.R_FRIENDLY_FIRE
                        self._stats["friendly_fire"] += 1
                    self.cooldown = self.COOLDOWN
                # cooldown>0 ise ates olmaz, sadece slew cezasi alindi
        # action == MAX_TARGETS -> bekle (ek islem yok)

        # --- 2) Zamani ilerlet ---
        if self.cooldown > 0:
            self.cooldown -= 1

        for t in self.slots:
            if t is None or not t["active"]:
                continue
            t["distance"] -= t["speed"]
            # slalom: temel kerteriz + sinus salinimi
            t["bearing"] = (t["base_bearing"]
                            + self.SLALOM_AMP * math.sin(self.SLALOM_FREQ * self.steps + t["phase"]))
            # savunma cizgisini gecti mi?
            if t["distance"] <= self.DEFENSE_RADIUS:
                if t["is_enemy"]:
                    reward += self.R_LEAK
                    self._stats["leaked"] += 1
                t["active"] = False  # dost gecerse sessizce cikar

        # --- 3) Zaman cezasi ---
        reward += self.R_TIME

        # --- 4) Bolum sonu ---
        any_active = any(t is not None and t["active"] for t in self.slots)
        terminated = not any_active
        truncated = self.steps >= self.MAX_STEPS

        info = dict(self._stats)
        return self._get_obs(), float(reward), terminated, truncated, info
