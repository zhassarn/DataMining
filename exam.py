# Task 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

full_name = "Zhassar Otan"

letters_count = len(full_name.replace(" ", ""))
N = letters_count * 17

print("Атыңыз (full_name):", full_name)
print("Әріп саны (бос орынсыз):", letters_count)
print("N = әріп саны * 17 =", N)

np.random.seed(42)

data = np.random.lognormal(mean=8.2, sigma=0.45, size=N)

outlier_idx = np.random.choice(np.arange(N), size=4, replace=False)
data[outlier_idx] *= np.random.uniform(5, 9, size=4)

df = pd.DataFrame({"purchase_kzt": np.round(data, 0).astype(int)})

print(df)

df.info()

print(df.describe())

plt.figure()
plt.hist(df["purchase_kzt"], bins=25)
plt.title("Histogram: purchase_kzt")
plt.xlabel("purchase_kzt")
plt.ylabel("frequency")
plt.show()

plt.figure()
plt.boxplot(df["purchase_kzt"], vert=True)
plt.title("Boxplot: purchase_kzt")
plt.ylabel("purchase_kzt")
plt.show()

q1 = df["purchase_kzt"].quantile(0.25)
q3 = df["purchase_kzt"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[(df["purchase_kzt"] < lower) | (df["purchase_kzt"] > upper)]
skewness = df["purchase_kzt"].skew()

print("\n--- Outlier (IQR method) ---")
print(f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
print(f"Lower bound={lower:.0f}, Upper bound={upper:.0f}")
print("Outlier саны =", len(outliers))
print("Ең үлкен outlier мәндері:\n", outliers.sort_values("purchase_kzt", ascending=False).head())

print("\n--- Skewness ---")
print("Skewness =", skewness)

# Task 2
from scipy import stats

x = df["purchase_kzt"].to_numpy()
n = len(x)
half = n // 2

group1 = x[:half]
group2 = x[half:]

print("n =", n, "| group1 =", len(group1), "| group2 =", len(group2))
print("Mean(group1) =", np.mean(group1))
print("Mean(group2) =", np.mean(group2))

print("\nH0: μ1 = μ2  (екі топтың орташа мәндері тең)")
print("H1: μ1 ≠ μ2  (екі топтың орташа мәндері тең емес)")

t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

print("\n--- Welch t-test нәтижесі ---")
print("t-statistic =", t_stat)
print("p-value     =", p_value)

alpha = 0.05
if p_value < alpha:
    decision = "H0 теріске шығарылады (орташа мәндер арасында статистикалық айырмашылық бар)"
else:
    decision = "H0 қабылданады / теріске шығарылмайды (статистикалық айырмашылық дәлелденбеді)"

print("\nα =", alpha)
print("Қорытынды:", decision)

# Task 3
from scipy.stats import chi2_contingency

counts = {
    ("Beginner", "Rare"): 12,
    ("Beginner", "Weekly"): 10,
    ("Beginner", "Daily"): 3,

    ("Intermediate", "Rare"): 6,
    ("Intermediate", "Weekly"): 12,
    ("Intermediate", "Daily"): 7,

    ("Advanced", "Rare"): 2,
    ("Advanced", "Weekly"): 8,
    ("Advanced", "Daily"): 10,
}

rows = []
for (level, freq), n in counts.items():
    rows += [{"experience_level": level, "play_frequency": freq} for _ in range(n)]

df_cat = pd.DataFrame(rows)

print("N бақылау саны =", len(df_cat))
print(df_cat.head())

ct = pd.crosstab(df_cat["experience_level"], df_cat["play_frequency"])
print("\n--- Contingency table (тәжірибе деңгейі × жиілік) ---")
print(ct)

chi2, p, dof, expected = chi2_contingency(ct)

print("\n--- Chi-square test нәтижесі ---")
print("chi2 statistic =", chi2)
print("p-value        =", p)
print("degrees of freedom =", dof)

expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
print("\n--- Expected frequencies (күтілетін жиіліктер) ---")
print(expected_df.round(2))

alpha = 0.05
if p < alpha:
    print(f"\nҚорытынды: p < {alpha}. H0 теріске шығарылады → айнымалылар арасында статистикалық тәуелділік БАР.")
else:
    print(f"\nҚорытынды: p ≥ {alpha}. H0 теріске шығарылмайды → тәуелділік дәлелденбеді (тәуелсіз болуы мүмкін).")