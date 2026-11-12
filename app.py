"""Dashboard Streamlit - Proyecto Integrador U4 (INC-2505)
Ejecutar:  streamlit run app.py
"""
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Invernadero - Dashboard", layout="wide")
sns.set_theme(style="whitegrid")


# ---------- Fase 1: datos deterministas en DuckDB ----------
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("SELECT setseed(0.42)")
    con.execute("""
        CREATE TABLE base AS
        SELECT i AS parcela_id,
               ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
               ROUND(50.0  + random() * 100.0, 2) AS fertilizante_kg,
               CASE WHEN random() < 0.50 THEN 'Tratamiento_A'
                    ELSE 'Tratamiento_B' END AS tipo_tratamiento,
               random() * 1.6 - 0.8 AS ruido,
               ROUND(18.0 + random() * 14.0, 2) AS temperatura
        FROM range(1, 101) t(i)""")
    return con.execute("""
        SELECT parcela_id, volumen_riego, fertilizante_kg, tipo_tratamiento, temperatura,
               ROUND(5.0 + 0.035 * volumen_riego + 0.020 * fertilizante_kg
                     + CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END
                     + ruido, 2) AS rendimiento_ton
        FROM base ORDER BY parcela_id""").fetchdf()


df_all = cargar_datos()

# ---------- Filtros dinámicos (Hito 4.2) ----------
st.sidebar.header("Filtros")
trat = st.sidebar.multiselect("Tratamiento", ["Tratamiento_A", "Tratamiento_B"],
                              default=["Tratamiento_A", "Tratamiento_B"])
r_min, r_max = st.sidebar.slider("Volumen de riego (m³/ha)", 100.0, 300.0, (100.0, 300.0))
df = df_all[df_all.tipo_tratamiento.isin(trat)
            & df_all.volumen_riego.between(r_min, r_max)].copy()

st.title("🌱 Optimización del rendimiento agrícola y eficiencia de riego")
st.caption("Invernaderos inteligentes de jitomate/aguacate · INC-2505")

if len(df) < 10:
    st.warning("Pocos datos con estos filtros (mínimo 10 parcelas).")
    st.stop()

# ---------- Fase 2: modelo OLS ----------
X = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]])
modelo = sm.OLS(df["rendimiento_ton"], X).fit()
df["residuo"] = df["rendimiento_ton"] - modelo.predict(X)

# ---------- Fase 3: prueba t (cola derecha) ----------
a = df.loc[df.tipo_tratamiento == "Tratamiento_A", "rendimiento_ton"]
b = df.loc[df.tipo_tratamiento == "Tratamiento_B", "rendimiento_ton"]
p_val, ic, dif = None, None, None
if len(a) > 1 and len(b) > 1:
    _, p_val = stats.ttest_ind(b, a, equal_var=True, alternative="greater")
    dif = b.mean() - a.mean()
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    ee = np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    tc = stats.t.ppf(0.975, len(a) + len(b) - 2)
    ic = (dif - tc * ee, dif + tc * ee)

# ---------- Fase 5: tarjetas KPI ----------
k1, k2, k3 = st.columns(3)
k1.metric("Rendimiento promedio global", f"{df.rendimiento_ton.mean():.2f} ton/ha")
k2.metric("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.1f} %")
k3.metric("P-valor de la prueba t", "—" if p_val is None else f"{p_val:.2e}")

if p_val is not None:
    if p_val < 0.05:
        st.success(f"p < 0.05 → se rechaza H0: Tratamiento_B aumenta el rendimiento "
                   f"(+{dif:.2f} ton/ha, IC95% [{ic[0]:.2f}, {ic[1]:.2f}]). "
                   "Se recomienda autorizar la inversión.")
    else:
        st.error("p ≥ 0.05 → no se rechaza H0: sin evidencia para escalar el Tratamiento_B.")
else:
    st.info("Selecciona ambos tratamientos para ver la prueba t.")

# ---------- Fase 4 + 5: cuadrícula 2x2 ----------
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

with c1:
    st.subheader("1. Mapa de calor (Pearson)")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    corr = df[["volumen_riego", "fertilizante_kg", "temperatura", "rendimiento_ton"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

with c2:
    st.subheader("2. Dispersión + recta OLS")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.regplot(data=df, x="volumen_riego", y="rendimiento_ton", ci=95,
                scatter_kws={"alpha": .6}, line_kws={"color": "red"}, ax=ax)
    ax.set_xlabel("Volumen de riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)

with c3:
    st.subheader("3. Boxplot por tratamiento")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton",
                hue="tipo_tratamiento", legend=False, ax=ax)
    sns.stripplot(data=df, x="tipo_tratamiento", y="rendimiento_ton",
                  color="k", alpha=.3, ax=ax)
    ax.set_xlabel("Tipo de tratamiento")
    ax.set_ylabel("Rendimiento (ton/ha)")
    st.pyplot(fig)

with c4:
    st.subheader("4. Residuos del modelo")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.histplot(df["residuo"], kde=True, bins=15, ax=ax)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Residuo = Y real − Y predicho (ton/ha)")
    st.pyplot(fig)

with st.expander("Ver tabla de datos y resumen OLS"):
    st.dataframe(df, width="stretch")
    st.text(modelo.summary().as_text())
