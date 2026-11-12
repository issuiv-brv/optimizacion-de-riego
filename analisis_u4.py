"""Proyecto Integrador U4 - INC-2505
Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego
Fases 1 a 4 (sin interfaz): imprime resultados y guarda las 4 gráficas.
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- FASE 1: DuckDB determinista ----------------
con = duckdb.connect(":memory:")            # BD en memoria
con.execute("SELECT setseed(0.42)")         # semilla global

# Tabla base: predictores y asignación aleatoria de tratamiento
con.execute("""
CREATE TABLE base AS
SELECT
    i                                         AS parcela_id,
    ROUND(100.0 + random() * 200.0, 2)        AS volumen_riego,     -- 100-300 m3/ha
    ROUND(50.0  + random() * 100.0, 2)        AS fertilizante_kg,   -- 50-150 kg/ha
    CASE WHEN random() < 0.50 THEN 'Tratamiento_A'
         ELSE 'Tratamiento_B' END             AS tipo_tratamiento,
    random() * 1.6 - 0.8                      AS ruido,             -- -0.8 a +0.8
    ROUND(18.0 + random() * 14.0, 2)          AS temperatura        -- solo para el heatmap
FROM range(1, 101) t(i)
""")

# Tabla final: rendimiento = b0 + b1*riego + b2*fert + efecto + ruido
con.execute("""
CREATE TABLE telemetria_invernadero AS
SELECT
    parcela_id, volumen_riego, fertilizante_kg, tipo_tratamiento, temperatura,
    ROUND(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
          + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
          + ruido, 2)                         AS rendimiento_ton
FROM base
ORDER BY parcela_id
""")

# Fase 1.3: extracción SQL -> pandas
df = con.execute("SELECT * FROM telemetria_invernadero").fetchdf()
print("=== FASE 1: dataset ===")
print(df.head(10).to_string(index=False))
print(f"\nN = {len(df)} | rendimiento min/max = "
      f"{df.rendimiento_ton.min():.2f} / {df.rendimiento_ton.max():.2f} ton/ha")
print(con.execute("""SELECT tipo_tratamiento, COUNT(*) n,
      ROUND(AVG(rendimiento_ton),2) media FROM telemetria_invernadero
      GROUP BY 1 ORDER BY 1""").fetchdf().to_string(index=False))

# ---------------- FASE 2: Correlación y OLS ----------------
print("\n=== FASE 2: correlación y regresión ===")
cols = ["volumen_riego", "fertilizante_kg", "temperatura", "rendimiento_ton"]
corr = df[cols].corr(method="pearson")
print(corr.round(3))

X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
y = df["rendimiento_ton"]
modelo = sm.OLS(y, X).fit()
print(modelo.summary())
rmse = float(np.sqrt(np.mean(modelo.resid ** 2)))
print(f"R2 ajustado = {modelo.rsquared_adj:.4f} | RMSE = {rmse:.4f}")
df["residuo"] = y - modelo.predict(X)        # Y_real - Y_predicho

# ---------------- FASE 3: Prueba t ----------------
print("\n=== FASE 3: inferencia ===")
a = df.loc[df.tipo_tratamiento == "Tratamiento_A", "rendimiento_ton"]
b = df.loc[df.tipo_tratamiento == "Tratamiento_B", "rendimiento_ton"]
# H0: mu_B - mu_A <= 0   vs   H1: mu_B - mu_A > 0  (cola derecha)
t_stat, p_val = stats.ttest_ind(b, a, equal_var=True, alternative="greater")
dif = b.mean() - a.mean()
na, nb = len(a), len(b)
sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
ee = np.sqrt(sp2 * (1 / na + 1 / nb))
tcrit = stats.t.ppf(0.975, na + nb - 2)
ic = (dif - tcrit * ee, dif + tcrit * ee)
print(f"media A = {a.mean():.3f} (n={na}) | media B = {b.mean():.3f} (n={nb})")
print(f"diferencia (B-A) = {dif:.3f} ton/ha")
print(f"t = {t_stat:.3f} | p-valor (una cola) = {p_val:.3e}")
print(f"IC 95% de la diferencia = ({ic[0]:.3f}, {ic[1]:.3f})")
if p_val < 0.05:
    print("VEREDICTO: p < 0.05 -> se RECHAZA H0. Tratamiento_B incrementa el "
          "rendimiento; se recomienda autorizar la inversión.")
else:
    print("VEREDICTO: p >= 0.05 -> no se rechaza H0. No hay evidencia para escalar B.")

# ---------------- FASE 4: Gráficas ----------------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0, 0])
ax[0, 0].set_title("1. Mapa de calor de correlación (Pearson)")

sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ci=95,
            scatter_kws={"alpha": .6}, line_kws={"color": "red"}, ax=ax[0, 1])
ax[0, 1].set_title("2. Dispersión + recta OLS (banda 95%)")
ax[0, 1].set_xlabel("Volumen de riego (m³/ha)")
ax[0, 1].set_ylabel("Rendimiento (ton/ha)")

sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", hue="tipo_tratamiento",
            legend=False, ax=ax[1, 0])
sns.stripplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", color="k",
              alpha=.3, ax=ax[1, 0])
ax[1, 0].set_title(f"3. Boxplot por tratamiento (p={p_val:.2e})")
ax[1, 0].set_xlabel("Tipo de tratamiento")
ax[1, 0].set_ylabel("Rendimiento (ton/ha)")

sns.histplot(df["residuo"], kde=True, bins=15, ax=ax[1, 1])
ax[1, 1].axvline(0, color="red", ls="--")
ax[1, 1].set_title("4. Histograma de residuos + KDE")
ax[1, 1].set_xlabel("Residuo (ton/ha)")

plt.tight_layout()
plt.savefig("graficas_u4.png", dpi=110)
print("\nGráficas guardadas en graficas_u4.png")
