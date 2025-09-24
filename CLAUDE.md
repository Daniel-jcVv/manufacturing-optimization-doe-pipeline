# DOE Tool Optimization - End-to-End Implementation Guide

Este documento te guía a construir el proyecto end-to-end hoy, como portfolio. Todas las implementaciones de código tendrán nombres, variables y comentarios en inglés.

## Orden de implementación (alto nivel)

1) Infraestructura local mínima (Docker opcional) y entorno Python
2) Datos sintéticos y utilidades base
3) Ingestión y procesamiento batch mínimo
4) Análisis DOE (Yates) y cálculos de ahorro
5) Carga a warehouse (PostgreSQL) y modelos dbt (stub)
6) Orquestación con Airflow (DAG mínimo funcional)
7) Dashboard Streamlit mínimo
8) Modelos ML (stub entrenable) y MLflow (opcional)
9) Pruebas básicas (unit/integration) y Makefile comandos

## Paso a paso (detalle)

- Paso 1: Entorno
  - Crear `requirements.txt` (si no existe) con: pandas, numpy, psycopg2-binary, sqlalchemy, airflow (si usas local), streamlit, plotly, scikit-learn.
  - Crear `.env` con credenciales de Postgres y rutas de datos.

- Paso 2: Datos y utilidades
  - Crear `src/utils/config.py` para leer `.env`.
  - Crear `src/utils/database.py` con conexión SQLAlchemy a Postgres.
  - Crear `src/ingestion/processing/data_simulator.py` que genere `data/raw/*.csv`.

- Paso 3: ETL batch mínimo
  - Crear `src/processing/etl/extract.py`, `transform.py`, `load.py` con funciones puras y docstrings.

- Paso 4: DOE
  - Crear `src/processing/doe/yates_algorithm.py` con clase `YatesAlgorithm` y método `find_optimal_conditions`.
  - Crear `src/processing/doe/costs.py` para calcular métricas de ahorro y CPU.

- Paso 5: Warehouse y dbt
  - Crear `sql/create_tables.sql` con tablas `experiment_results`, `production_data`, `cost_savings`.
  - En `dbt/` dejar `dbt_project.yml` y un modelo `models/cost_savings.sql` (stub).

- Paso 6: Airflow
  - Crear `airflow/dags/doe_pipeline.py` que:
    - genera datos -> analiza DOE -> calcula ahorros -> carga a Postgres.

- Paso 7: Dashboard
  - Crear `dashboard/app.py` que lea Postgres y muestre KPIs básicos.

- Paso 8: ML (opcional hoy)
  - Crear `src/ml_models/tool_life_predictor.py` (stub) con entrenamiento simple RF.

- Paso 9: Tests y Makefile
  - Tests unit para Yates y costos en `tests/`.
  - `Makefile` con comandos: setup, start-db, doe-analysis, dashboard, test.

## Árbol sugerido (respetando carpetas existentes)

- `src/utils/config.py`
- `src/utils/database.py`
- `src/ingestion/processing/data_simulator.py`
- `src/processing/etl/extract.py`
- `src/processing/etl/transform.py`
- `src/processing/etl/load.py`
- `src/processing/doe/yates_algorithm.py`
- `src/processing/doe/costs.py`
- `sql/create_tables.sql`
- `airflow/dags/doe_pipeline.py`
- `dashboard/app.py`
- `src/ml_models/tool_life_predictor.py`
- `tests/test_yates.py`
- `tests/test_costs.py`
- `Makefile`

## Reglas de estilo de código

- Nombres en inglés, descriptivos. Sin abreviaturas crípticas.
- Docstrings y comentarios explicando el propósito (what/why).
- Funciones puras donde sea posible; errores manejados con mensajes claros.

## Cómo presentar en el CV

- Enfatiza: arquitectura E2E, DOE real, ahorro proyectado, dashboard y orquestación.
- KPIs: CPU -18.75%, vida útil +275%, ROI 425% (simulado).
- Enlace a repositorio, capturas del dashboard y diagrama de arquitectura. 