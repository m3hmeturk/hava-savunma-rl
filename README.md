# Hava Savunma Sistemi için Pekiştirmeli Öğrenme ile Hedef Önceliklendirme

Bu proje, bir hava savunma senaryosunda **çok hedefli önceliklendirme** problemini
pekiştirmeli öğrenme (Reinforcement Learning, RL) ile çözmeyi amaçlar. İki farklı RL
paradigması (**PPO** ve **DQN**) eğitilip, klasik bir karar-analizi sezgiseli (**AHP**)
ve bir **rastgele** taban çizgisi ile karşılaştırılır.

> Pekiştirmeli Öğrenme dersi dönem projesi.

---

## 1. Problem Tanımı

İki boyutlu, basit kinematik bir simülasyonda savunma bölgesi merkezde (0,0)
konumlanır. Hedefler karşıdan, sabit hızla, yavaşça ve **slalom** (yanal sinüs
salınımı) yaparak yaklaşır. Taret her adımda hangi hedefe yöneleceğine /
ateşleyeceğine karar verir.

**Amaç:** Yüksek tehditli **düşman** hedefleri savunma çizgisini geçmeden
etkisizleştirmek; **dost** hedeflere asla ateş etmemek; tehdide göre doğru
önceliklendirme yapmak.

---

## 2. MDP Formülasyonu

Problem bir Markov Karar Süreci (MDP) olarak modellenmiştir.

### Durum (State) — 47 boyutlu vektör
En fazla **5 hedef slotu**, her hedef için **9 özellik** (hepsi `[-1, 1]` aralığına
normalize) + **2 taret durumu**:

| # | Özellik | Açıklama |
|---|---------|----------|
| 1 | aktif | slot dolu/boş |
| 2 | mesafe | savunma bölgesine uzaklık |
| 3 | hız | yaklaşma hızı |
| 4 | kerteriz | açısal yön (işaretli) |
| 5 | tehdit | tehdit seviyesi (1–3) |
| 6 | düşman mı | dost/düşman bayrağı |
| 7 | takip hazır | otonom takip menzilinde mi |
| 8 | vuruş hazır | otonom vuruş menzilinde mi |
| 9 | erişim süresi | mesafe / hız |

Taret durumu: mevcut nişan açısı + cooldown.
Toplam: `9 × 5 + 2 = 47`.

### Aksiyon (Action) — Discrete(6)
- `0–4`: ilgili hedef slotuna yönel/ateşle
- `5`: bekle / mevcut yönelimi koru

### Ödül (Reward)

| Olay | Ödül |
|------|------|
| Düşmanı çizgiyi geçmeden etkisizleştir | `+1.0 × tehdit seviyesi` |
| Dosta ateş et | `−2.0` |
| Düşman çizgiyi geçer (sızıntı) | `−1.0` |
| Her adım (zaman cezası) | `−0.01` |
| Büyük açı değişimi (taret dönüşü) | `−0.05 × normalize açı farkı` |
| Geçersiz seçim (boş/hazır olmayan hedef) | `−0.02` |

### Bölüm (Episode)
- Bölüm başına 3–5 hedef (~%70 düşman, %30 dost)
- Maksimum 300 adım; tüm hedefler çözülünce ya da süre dolunca biter

---

## 3. Yöntemler

| Yöntem | Tür | Açıklama |
|--------|-----|----------|
| **Rastgele** | Taban çizgisi | Aksiyon uzayından rastgele seçim (alt sınır) |
| **AHP** | Klasik karar analizi | İkili karşılaştırma matrisinden ağırlık çıkarıp en yüksek öncelikli düşmanı seçer |
| **PPO** | RL — politika-gradyan, on-policy | Proximal Policy Optimization |
| **DQN** | RL — değer-tabanlı, off-policy | Deep Q-Network |

### AHP (Analitik Hiyerarşi Süreci)
Önceliklendirme kriterleri **Tehdit**, **Yakınlık**, **Hız** olarak alınmış;
dost/düşman ve vuruşa hazır olma birer **sert filtre** (uygunluk şartı) olarak
uygulanmıştır. Saaty 1–9 skalasıyla kurulan ikili karşılaştırma matrisinden
elde edilen ağırlıklar:

- Tehdit: **0.648**, Yakınlık: **0.230**, Hız: **0.122**
- Tutarlılık Oranı (CR): **0.0032** (< 0.10, tutarlı)

### PPO ve DQN
Stable-Baselines3 ile eğitilmiştir. Her iki yöntemde de ajanın "pasiflik" lokal
optimumuna saplanmaması için keşif teşviki kullanılmıştır:
- PPO: `ent_coef = 0.01` (entropi bonusu)
- DQN: `exploration_fraction = 0.5` (uzun epsilon-greedy keşfi)

Adil karşılaştırma için ikisi de **300.000 adım** eğitilir.

---

## 4. Proje Yapısı

```
hava-savunma-rl/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ train.py                 # PPO eğitimi
├─ train_dqn.py             # DQN eğitimi
├─ evaluate.py              # tüm ajanları değerlendirir, grafik+tablo üretir
├─ test_env.py              # ortam SB3 uyum testi
├─ test_baselines.py        # baseline ajan testi
├─ src/
│  ├─ env/
│  │  └─ target_env.py      # özel Gymnasium ortamı (gym.Env)
│  └─ agents/
│     └─ baselines.py       # Rastgele + AHP ajanları
├─ data/                    # test senaryoları / trajektoriler
├─ results/                 # modeller, grafikler, tablo (çıktılar)
├─ logs/                    # TensorBoard + monitor kayıtları
└─ report/                  # rapor ve sunum
```

---

## 5. Kurulum

Python 3.11+ gerekir.

```bash
# Sanal ortam
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Bağımlılıklar
pip install -r requirements.txt
```

`requirements.txt`:
```
stable-baselines3[extra]
gymnasium
numpy
pandas
matplotlib
```

---

## 6. Kullanım

Aşağıdaki komutlar proje **kök dizininden** çalıştırılır.

```bash
# 1) Ortamın çalıştığını doğrula
python test_env.py
python test_baselines.py

# 2) RL ajanlarını eğit (her biri ~5 dk)
python train.py        # PPO  -> results/ppo_model.zip + learning_curve.png
python train_dqn.py    # DQN  -> results/dqn_model.zip + learning_curve_dqn.png

# 3) Tüm yöntemleri karşılaştır (grafik + tablo üretir)
python evaluate.py

# 4) (Opsiyonel) Eğitimi TensorBoard ile izle
tensorboard --logdir logs
```

`evaluate.py` `results/` klasörüne eğitilmiş modelleri otomatik bulup tabloya
ekler; PPO ya da DQN modeli yoksa sadece mevcut ajanlarla çalışır.

### Üretilen çıktılar (`results/`)
- `comparison_table.csv` — sayısal karşılaştırma tablosu
- `reward_comparison.png` — ortalama ödül (çubuk grafik)
- `metrics_comparison.png` — etkisizleştirme / sızıntı / dost ateşi
- `learning_curve.png`, `learning_curve_dqn.png` — bireysel öğrenme eğrileri
- `learning_curves_combined.png` — PPO ve DQN öğrenme eğrileri tek eksende

---

## 7. Sonuçlar

200 test bölümü üzerinden (sabit senaryolar, tüm ajanlar aynı girdiyi görür):

| Yöntem | Ort. Ödül | Etkisizleştirme | Sızıntı | Dost Ateşi |
|--------|-----------|-----------------|---------|------------|
| Rastgele | −28.1 | 556 | 0 | 2913 |
| **AHP** | **+3.69** | 556 | 0 | 0 |
| PPO | +1.23 | 435 | 121 | 0 |
| DQN | −1.20 | 281 | 275 | 0 |

**Bulgular:**
- Her iki RL yöntemi de dost/düşman ayrımını tam öğrendi (0 dost ateşi).
- **PPO (on-policy)**, **DQN'den (off-policy)** daha hızlı ve kararlı öğrenerek
  AHP referansına yaklaştı; DQN daha düşük seviyede plato yaptı.
- Hiçbir RL yöntemi elle ayarlı AHP sezgiselini geçemedi; bu, basit ve iyi
  tanımlı bir problemde uzman sezgiselinin güçlü bir taban çizgisi olduğunu
  gösterir.

---

## 8. Teknik Notlar

- Tüm ajanlar (RL ve klasik) **aynı 47 boyutlu gözlemden** karar verir; bu sayede
  karşılaştırma adildir.
- Değerlendirmede sabit `seed` kullanıldığı için sonuçlar tekrar üretilebilir.
- Ortam, Gymnasium API'sine uyumludur (`stable-baselines3.common.env_checker`
  ile doğrulanmıştır).

---

## 9. Bağımlılıklar

- [Gymnasium](https://gymnasium.farama.org/) — RL ortam arayüzü
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PPO/DQN
- NumPy, pandas, matplotlib
