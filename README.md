# Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego

Proyecto Integrador U4 · Análisis y Exploración de Datos (INC-2505)

Dashboard en Streamlit con telemetría simulada de invernadero (DuckDB, `setseed(0.42)`, N=100):
correlación de Pearson, regresión OLS (statsmodels), prueba t de dos muestras con IC 95 % (scipy)
y 4 gráficas con filtros dinámicos.

## Ejecutar localmente
```
pip install -r requirements.txt
streamlit run app.py
```

## Archivos
- `app.py` – dashboard (punto de entrada para Streamlit Community Cloud)
- `analisis_u4.py` – análisis en consola (Fases 1 a 4)
- `requirements.txt` – dependencias fijadas
