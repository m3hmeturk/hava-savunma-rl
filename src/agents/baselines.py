"""
Baseline Ajanlar: Rastgele + AHP Sezgisel
=========================================

Her iki ajan da PPO ile AYNI 47 boyutlu gozlem vektorunden karar verir;
boylece karsilastirma adil olur (hepsi ayni girdiyi gorur).

Gozlem duzeni (env._get_obs ile birebir ayni):
  Her slot (9 deger): [aktif, mesafe/100, hiz/0.6, kerteriz/(pi/2),
                       tehdit/3, dusman, takip_hazir, vurus_hazir, ttr/300]
  Son 2 deger        : [nisan_acisi/(pi/2), cooldown/3]
"""

import numpy as np

MAX_TARGETS = 5
FEATURES_PER_TARGET = 9


# ----------------------------------------------------------------------
# Gozlem vektorunu hedef ozelliklerine ayristirir
# ----------------------------------------------------------------------
def decode_observation(obs):
    """47 boyutlu gozlemi, her hedef icin bir sozluk listesine cevirir."""
    targets = []
    for i in range(MAX_TARGETS):
        base = i * FEATURES_PER_TARGET
        targets.append({
            "active":      obs[base + 0] > 0.5,
            "distance":    obs[base + 1],   # 0..1 (yakinlik = 1 - bu)
            "speed":       obs[base + 2],   # 0..1
            "bearing":     obs[base + 3],
            "threat":      obs[base + 4],   # 0..1 (tehdit/3)
            "is_enemy":    obs[base + 5] > 0.5,
            "track_ready": obs[base + 6] > 0.5,
            "fire_ready":  obs[base + 7] > 0.5,
            "ttr":         obs[base + 8],
        })
    cooldown = obs[-1]  # 0..1
    return targets, cooldown


# ----------------------------------------------------------------------
# AHP agirliklarini ikili karsilastirma matrisinden hesaplar
# ----------------------------------------------------------------------
def compute_ahp_weights():
    """
    Kriterler: [Tehdit, Yakinlik, Hiz]
    Saaty 1-9 skalasi ile ikili karsilastirma matrisi.
    Donen: (agirliklar, CR)
    """
    A = np.array([
        [1,     3,   5],
        [1/3,   1,   2],
        [1/5,   1/2, 1],
    ])
    n = A.shape[0]
    eigvals, eigvecs = np.linalg.eig(A)
    idx = np.argmax(eigvals.real)
    w = eigvecs[:, idx].real
    w = w / w.sum()
    lam_max = eigvals.real[idx]
    CI = (lam_max - n) / (n - 1)
    RI = 0.58  # n=3 icin Random Index
    CR = CI / RI
    return w, CR


# ----------------------------------------------------------------------
# Rastgele ajan (alt sinir)
# ----------------------------------------------------------------------
class RandomAgent:
    """Gecerli aksiyon uzayindan rastgele secim yapar."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


# ----------------------------------------------------------------------
# AHP sezgisel ajan (klasik yontem)
# ----------------------------------------------------------------------
class AHPAgent:
    """
    Uygun (aktif + dusman + vurusa hazir) hedefler arasindan, AHP
    agirliklariyla en yuksek oncelik skorlu olani secer. Uygun hedef
    yoksa ya da taret cooldown'daysa bekler (aksiyon 5).
    """

    def __init__(self):
        self.weights, self.cr = compute_ahp_weights()  # [Tehdit, Yakinlik, Hiz]

    def score(self, t):
        threat    = t["threat"]            # yuksek = onemli
        proximity = 1.0 - t["distance"]    # yakin = onemli
        speed     = t["speed"]             # hizli = onemli
        return (self.weights[0] * threat
                + self.weights[1] * proximity
                + self.weights[2] * speed)

    def act(self, obs):
        targets, cooldown = decode_observation(obs)
        if cooldown > 1e-6:          # taret hazir degil -> bekle
            return MAX_TARGETS       # aksiyon 5

        best_idx, best_score = -1, -np.inf
        for i, t in enumerate(targets):
            # sert filtre: aktif + dusman + vurusa hazir
            if t["active"] and t["is_enemy"] and t["fire_ready"]:
                s = self.score(t)
                if s > best_score:
                    best_score, best_idx = s, i

        return best_idx if best_idx >= 0 else MAX_TARGETS  # uygun yoksa bekle
