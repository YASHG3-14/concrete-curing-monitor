"""
Simple concrete strength prediction (UCI data + synthetic maturity data).
Runs in a few seconds.

Idea in 4 steps:
 1. Learn "mix + age -> strength" from the UCI dataset.
 2. Make synthetic slab data: random curing temperature and age -> maturity
    (Nurse-Saul) -> equivalent age -> strength from step 1 (+ noise).
 3. Train a Random Forest on mix + maturity.
 4. Compare with the classic maturity curve: strength = a + b*ln(maturity).

pip install numpy pandas scikit-learn ucimlrepo xlrd openpyxl
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
MIX = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse", "fine"]

# ---- 1. Load UCI data -------------------------------------------------------
try:
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=165)
    uci = pd.concat([d.data.features, d.data.targets], axis=1)
except Exception:
    try:
        uci = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                            "concrete/compressive/Concrete_Data.xls")
    except Exception:   # offline: put the downloaded file next to this script
        uci = pd.read_csv("Concrete_Data.csv")
uci = uci.iloc[:, :9]
uci.columns = MIX + ["age", "strength"]          # age is in days
print("UCI samples:", len(uci))

# Prior model: mix + age -> strength
prior = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
prior.fit(uci[MIX + ["age"]], uci["strength"])

# ---- 2. Synthetic slab data -------------------------------------------------
N = 5000
syn = uci.sample(N, replace=True, random_state=1)[MIX].reset_index(drop=True)
temp = rng.uniform(10, 40, N)                    # average curing temperature (C)
age = rng.uniform(1, 28, N)                      # real age in days
T0 = -10                                         # Nurse-Saul datum temperature
syn["maturity"] = (temp - T0) * age * 24         # deg C * hours
eq_age = syn["maturity"] / ((20 - T0) * 24)      # equivalent age at 20 C (days)
syn["strength"] = prior.predict(syn[MIX].assign(age=eq_age.clip(lower=1)))
syn["strength"] += rng.normal(0, 1.5, N)         # test noise

# ---- 3. Train / test --------------------------------------------------------
train, test = train_test_split(syn, test_size=0.2, random_state=0)
features = MIX + ["maturity"]

rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
rf.fit(train[features], train["strength"])
pred_rf = rf.predict(test[features])

# ---- 4. Baseline: strength = a + b*ln(maturity) -----------------------------
b, a = np.polyfit(np.log(train["maturity"]), train["strength"], 1)
pred_base = a + b * np.log(test["maturity"])

print(f"\nMaturity curve : MAE={mean_absolute_error(test.strength, pred_base):.2f} MPa"
      f"  R2={r2_score(test.strength, pred_base):.3f}")
print(f"Random Forest  : MAE={mean_absolute_error(test.strength, pred_rf):.2f} MPa"
      f"  R2={r2_score(test.strength, pred_rf):.3f}")

# ---- Which inputs matter? ---------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:\n", imp.round(3).to_string())

# ---- Charts (saved to strength_charts.png) ----------------------------------
import matplotlib
matplotlib.use("Agg")                            # remove this line to pop up a window
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#888888"
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# 1) Predicted vs actual: points close to the dashed line = good model
lim = [0, max(test.strength.max(), pred_rf.max()) * 1.05]
ax[0].scatter(test.strength, pred_base, s=6, alpha=0.3, color=ORANGE, label="Maturity curve")
ax[0].scatter(test.strength, pred_rf, s=6, alpha=0.3, color=BLUE, label="Random Forest")
ax[0].plot(lim, lim, "--", color=GREY, lw=1, label="Perfect prediction")
ax[0].set(title="Predicted vs actual strength", xlabel="Actual (MPa)", ylabel="Predicted (MPa)")
ax[0].legend(markerscale=2, frameon=False)

# 2) Strength vs maturity: data, baseline curve, and RF trend
ax[1].scatter(test.maturity, test.strength, s=6, alpha=0.25, color=GREY, label="Test data")
m = np.linspace(test.maturity.min(), test.maturity.max(), 200)
ax[1].plot(m, a + b * np.log(m), color=ORANGE, lw=2, label="Maturity curve")
avg_mix = pd.DataFrame([train[MIX].median()] * 200).reset_index(drop=True)
avg_mix["maturity"] = m                          # vary only maturity, keep the mix fixed
ax[1].plot(m, rf.predict(avg_mix[features]), color=BLUE, lw=2, label="Random Forest (median mix)")
ax[1].set(title="Strength vs maturity", xlabel="Maturity (deg C x hours)", ylabel="Strength (MPa)")
ax[1].legend(frameon=False)

# 3) Feature importance: what the model relies on
imp.sort_values().plot.barh(ax=ax[2], color=BLUE)
ax[2].set(title="Feature importance", xlabel="Importance")

for x in ax:
    x.grid(alpha=0.2)
    x.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("strength_charts.png", dpi=150)
print("\nSaved chart: strength_charts.png")

# ---- Predict one slab -------------------------------------------------------
slab = test[features].iloc[[0]]
print(f"\nExample: maturity={slab.maturity.iloc[0]:.0f} C.h -> "
      f"predicted {rf.predict(slab)[0]:.1f} MPa (actual {test.strength.iloc[0]:.1f})")
