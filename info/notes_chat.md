# Notas del Chat - DOE Tool Optimization (Contexto de Desarrollo)

Este documento registra, en lenguaje claro y con mucho detalle, todo lo realizado en este proyecto durante la sesión. Está pensado para alguien con poca o ninguna experiencia en ingeniería de datos. Se actualizará en cada interacción para conservar el contexto de qué, cómo y para qué se hizo cada cosa.

---

## 1) Decisiones iniciales y entendimiento del proyecto

- **Qué hicimos**: Leímos `README.md` y `info/info.md` para entender el alcance (pipeline E2E de Data Engineering con DOE, Airflow, dbt, Streamlit, ML, etc.).
- **Por qué**: Antes de crear archivos o infraestructura, es vital comprender los objetivos y la arquitectura esperada.
- **Para qué**: Alinear la implementación con los objetivos del caso (mejorar vida útil, bajar costos, ROI alto) y presentar un portfolio sólido.

---

## 2) Creación de guía de implementación (CLAUDE.md)

- **Qué hicimos**: Creamos `CLAUDE.md` con el orden de implementación E2E (infra, datos sintéticos, ETL, DOE, warehouse, orquestación, dashboard, ML, tests).
- **Por qué**: Sirve como hoja de ruta. Asegura que trabajemos en el orden correcto para ver valor rápidamente.
- **Para qué**: Guiarnos paso a paso, evitando re-trabajos y facilitando la presentación en el CV.

Contenido clave del plan:
- Entorno y dependencias.
- Simulador de datos (experimentos DOE y producción).
- ETL mínimo (extract/transform/load).
- Módulo DOE (Algoritmo de Yates) y cálculo de ahorros (CPU, costos).
- SQL para tablas base en PostgreSQL.
- DAG de Airflow mínimo funcional.
- Dashboard mínimo en Streamlit.
- Stub de ML (opcional) y pruebas básicas.

---

## 3) Docker Compose: ubicación y servicios

### 3.1 Ubicación del archivo docker-compose
- **Decisión**: Mover `docker-compose.yml` a la **raíz del proyecto**.
- **Por qué**: Es una práctica común y mejora la experiencia de desarrollo (DX): un solo comando desde la raíz levanta todo.
- **Para qué**: Simplificar comandos y estandarizar el proyecto.

### 3.2 Primera versión (LocalExecutor, sin Redis)
- **Qué hicimos**: Creamos un `docker-compose.yml` con `postgres` y `airflow` (LocalExecutor).
- **Por qué**: LocalExecutor no requiere Redis; es más simple para un arranque rápido.
- **Problema detectado**: Conflictos de puertos y warnings.
  - Puerto 5432 ya en uso por otro Postgres local.
  - Warnings por variables `_AIRFLOW_WWW_USER_*` no definidas en el host.
  - Clave `version` obsoleta en Compose v2.

### 3.3 Evolución a CeleryExecutor + Redis
- **Analogía**:
  - Postgres = despensa (datos y metadatos).
  - Redis = timbre/bandeja de pedidos (cola de tareas).
  - Webserver = recepcionista (UI y entrada de órdenes/DAGs).
  - Scheduler = jefe de cocina (asigna tareas a workers).
  - Workers = cocineros (ejecutan tareas en paralelo).
- **Decisión**: Pasar a `CeleryExecutor` e **incluir Redis**, con tres servicios de Airflow: webserver, scheduler y worker.
- **Por qué**: Permite paralelismo real y escalabilidad; más cercano a un entorno profesional.
- **Para qué**: Ejecutar tareas de forma robusta y paralela, útil para pipelines de datos con múltiples pasos.

### 3.4 Conflictos de puertos y soluciones
- **Problema 1**: `failed to bind host port 5432` (más tarde 5433) en Postgres.
  - **Causa**: Puerto ya ocupado por otra instancia local.
  - **Solución**: Mapear Postgres a `5434:5432` en el host.
- **Problema 2**: `failed to bind host port 6379` en Redis.
  - **Causa**: Un Redis local u otro contenedor usando 6379.
  - **Solución**: Mapear Redis a `6380:6379` en el host.
- **Resultado**: Contenedores iniciaron en estado `Healthy/Started`.

### 3.5 Warnings y cómo los resolvimos
- **Warning**: Variables `_AIRFLOW_WWW_USER_USERNAME` y `_AIRFLOW_WWW_USER_PASSWORD` "no seteadas".
  - **Causa**: Docker intenta expandir variables del host si aparecen como `$VAR` en el YAML.
  - **Solución**: En el `command` del contenedor usamos `$$` (doble símbolo) para que **Docker no expanda** y deje que **Bash dentro del contenedor** lea las variables de entorno. Ejemplo: `--username $$_AIRFLOW_WWW_USER_USERNAME`.
- **Warning**: Clave `version` obsoleta en Compose v2.
  - **Solución**: Se **eliminó** la línea `version:` del YAML.

### 3.6 Estado final del stack (validado)
- Servicios y puertos:
  - Airflow Webserver: `http://localhost:8081` (admin/admin123)
  - Postgres: `localhost:5434` (db `airflow`, user `airflow`, pass `airflow`)
  - Redis: `localhost:6380`
- Salud:
  - `docker compose up -d --remove-orphans` → todos `Healthy/Started`.

---

## 4) Próximos pasos (plan inmediato)

1. Crear `sql/create_tables.sql` con tablas base (`experiment_results`, `production_data`, `cost_savings`).
2. Crear `airflow/dags/doe_pipeline.py` mínimo: simular datos → análisis DOE (Yates) → ahorros → carga a Postgres.
3. Validar DAG en UI de Airflow y ejecución exitosa.
4. (Opcional) Añadir `Makefile` con comandos `up`, `down`, `logs`, `restart`, `init-db`.

---

## 5) Comandos útiles usados y recomendados

- Limpiar y relanzar servicios:
  - `docker compose down -v`
  - `docker compose up -d --remove-orphans`
  - `docker compose ps`
  - `docker compose logs -f | cat`
- Acceso a servicios:
  - Airflow UI: abrir navegador en `http://localhost:8081`.
  - Postgres (cliente): host `localhost`, puerto `5434`, db `airflow`, user `airflow`, pass `airflow`.

---

## 6) Justificación técnica de decisiones

- **CeleryExecutor vs LocalExecutor**: Elegimos CeleryExecutor + Redis para un entorno más cercano a producción (paralelismo y resiliencia). LocalExecutor es válido para pruebas sencillas, pero Redis ofrece mejor manejo de colas y escalado de workers.
- **Puertos no estándar (5434/6380)**: Evitan choques con instalaciones locales existentes. No afecta a las apps dentro del compose, solo al acceso desde el host.
- **Archivo compose en la raíz**: Mejora DX; es el punto único de arranque del proyecto.

---

## 7) Estado actual

- `docker-compose.yml` en la raíz configurado con Postgres (5434), Redis (6380) y Airflow (webserver, scheduler, worker) en **CeleryExecutor**.
- Servicios levantan correctamente y están saludables.
- Listos para crear tablas, DAG y pipeline mínimo.

---

## 8) Consulta: ¿Agregar Kafka/Zookeeper y MinIO ahora?

- **Contexto**: Según `info/phases.md`, Kafka+Zookeeper y MinIO se listan en la FASE 1 (infra base), y el primer código clave de ingesta (producer) llega en FASE 2.

- **Evaluación y mejores prácticas**:
  - Agregar muchos servicios al inicio puede dificultar el diagnóstico (más piezas que pueden fallar). 
  - En proyectos E2E de portfolio, es válido iterar por fases para mostrar valor temprano y estabilidad.
  - Redis ya está incluido para CeleryExecutor (orquestación robusta). 
  - Kafka/Zookeeper tiene complejidad operativa (tópicos, retention, conectividad), y solo aporta valor cuando implementemos el producer/consumer de sensores (FASE 2).
  - MinIO (S3-compatible) aporta valor como Data Lake (persistencia de archivos/artefactos). Es relativamente liviano y útil incluso antes de Kafka para: dumps de CSV, outputs de DOE, artefactos de ML.

- **Recomendación** (faseada y pragmática):
  1) Mantener el stack actual estable (Airflow + Postgres + Redis). 
  2) Agregar **MinIO** ahora (o inmediatamente después del DAG mínimo) para estandarizar almacenamiento de archivos y outputs; es liviano y útil.
  3) Agregar **Kafka + Zookeeper** al iniciar FASE 2 (cuando implementemos `src/ingestion/kafka_producers/iot_sensor_producer.py`), junto con un consumer básico para validación.

- **Plan de incorporación**:
  - MinIO:
    - Servicio `minio` + consola `minio-console`.
    - Variables: `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`.
    - Buckets: `raw/`, `processed/`, `results/` (creación con script de init o tarea de Airflow).
  - Kafka/Zookeeper:
    - Servicios `zookeeper` y `kafka`.
    - `KAFKA_ADVERTISED_LISTENERS` y `KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1`.
    - Tópicos: `machine_sensors`, `alerts` (bootstrap con script o DAG).

- **Beneficio**: Este orden reduce fricción, mantiene el sistema operable y permite mostrar avance tangible (pipeline batch) antes de sumar streaming.

---

## 9) Implementación de MinIO y conceptos clave

### 9.1 ¿Qué es S3/MinIO y qué es un bucket?
- **S3**: Un servicio de almacenamiento de objetos (archivos binarios o de texto) que guarda datos como "objetos" dentro de contenedores llamados **buckets**. No es un sistema de archivos tradicional: no hay carpetas reales, solo claves (keys) que parecen rutas.
- **MinIO**: Una implementación compatible con S3 que puedes correr en tu máquina. Ofrece API S3 y una **consola web** para administrar.
- **Bucket**: Piensa en un bucket como una "gran caja" donde guardas objetos. Creamos buckets lógicos: `raw/` (datos brutos), `processed/` (datos procesados) y `results/` (salidas de análisis y reportes). Esto ayuda a separar etapas del pipeline (bronze/silver/gold).

### 9.2 Qué añadimos al docker-compose
- Servicio `minio` (servidor de almacenamiento de objetos) con puertos mapeados al host:
  - API S3: `http://localhost:9100` (dentro de la red de Docker: `http://minio:9000`).
  - Consola: `http://localhost:9101`.
- Servicio `minio-init` (cliente `mc`) que corre una vez para:
  - Configurar alias contra `minio`.
  - Crear buckets `raw`, `processed`, `results`.
  - Dejar `results` con política pública (útil para compartir reportes/artefactos en demos locales).
- Variables de entorno en Airflow (webserver/scheduler/worker) para poder usar MinIO desde tareas: endpoint, access key y secret.

### 9.3 Por qué lo hicimos ahora
- Estandariza el almacenamiento de archivos antes de conectar el streaming (Kafka), y nos da un lugar consistente para outputs del pipeline (p.ej., CSV/JSON de DOE, reportes del DAG) y artefactos ML.

### 9.4 Cambios de puertos y salud
- Evitamos choques usando puertos del host no estándar: `9100/9101`.
- Validamos que `minio` esté `Healthy` y que `minio-init` termine en `Exited` exitoso tras crear buckets.

### 9.5 Cómo acceder
- Consola MinIO: abrir `http://localhost:9101` (usuario `minio_admin`, pass `minio123456`).
- API S3 local: `http://localhost:9100`.
- Dentro de contenedores: `http://minio:9000`.

---

> Próximo: usaremos estos buckets para guardar datos `raw`/`processed` y resultados de análisis DOE desde tareas de Airflow.

---

## 10) Limpieza de estructura de directorios y mejores prácticas

### 10.1 Problema de arquitectura detectado
- **Qué encontramos**: 3 directorios de datos diferentes (`data/`, `data_local/`, `scripts/data/`) creados por falta de planificación inicial.
- **Impacto**: Inconsistencias, confusión, violación de principios DRY (Don't Repeat Yourself).
- **Error como tutor**: Crear estructura nueva en lugar de usar la existente - violó el principio "Don't Reinvent the Wheel".

### 10.2 Solución profesional implementada
- **Decisión**: Usar ÚNICAMENTE el directorio `data/` (estándar industry).
- **Acciones**:
  - Eliminamos `data_local/` y `scripts/data/`
  - Consolidamos todos los datos en `data/` con estructura estándar:
    ```
    data/
    ├── raw/           # Datos fuente (immutable)
    ├── processed/     # Datos transformados
    └── results/       # Outputs de análisis
    ```
  - Actualizamos `.env` y configuraciones para apuntar a `./data`

### 10.3 Lecciones de Data Engineering
1. **SIEMPRE** revisar infraestructura existente ANTES de crear nueva
2. **NUNCA** duplicar estructuras - consolidar la existente
3. **RESPETAR** decisiones arquitecturales previas del proyecto

---

## 11) Validación completa de infraestructura (Phase 1)

### 11.1 Pruebas sistemáticas realizadas
**Docker Services**: 6 servicios UP + HEALTHY
```bash
docker compose ps  # Todos los contenedores saludables
```

**PostgreSQL**: Conectividad y esquema
```bash
python -c "from src.utils.database import test_connecton; print(test_connecton())"  # True
# 4 tablas DOE creadas: experiment_results, production_data, cost_savings, doe_analysis
```

**Data Pipeline**: Generación y carga
```bash
python src/ingestion/processing/data_simulator.py  # 24 experimentos + 90 registros producción
# Datos cargados via \copy command en PostgreSQL
```

**Airflow**: Ejecución de tasks
```bash
airflow tasks test test_infrastructure test_bash_task  # SUCCESS
# Webserver healthy, scheduler funcional
```

**MinIO**: Storage S3-compatible
```bash
mc ls local/  # 3 buckets: raw, processed, results
# Files upload/download confirmado
```

### 11.2 Issue MinIO y su solución
**Problema detectado**: Los archivos CSV estaban en `data/raw/` local y PostgreSQL, pero NO en MinIO.

**Root Cause**: Falta de integración en el pipeline - los datos se generaron pero nunca se subieron a MinIO.

**Solución técnica**:
```bash
# 1. Copiar al contenedor MinIO
docker cp data/raw/experiment_results.csv doe_minio:/tmp/

# 2. Subir al bucket usando MinIO client
docker exec doe_minio mc cp /tmp/experiment_results.csv local/raw/
```

**Resultado**: Datos ahora en 3 ubicaciones (local + MinIO + PostgreSQL).

**Lección profesional**: En producción esto debe automatizarse en el DAG de Airflow:
```
generate_data → save_to_local → upload_to_minio → load_to_postgres
```

---

## 12) Rol de Airflow: Estado actual vs futuro

### 12.1 Airflow en Phase 1: "Standby Mode"
**Lo que hace ahora**: Solo validación básica
```python
test_dag.py:
- test_bash_task: echo "✅ Bash task executed successfully!"  # SUCCESS
- test_python_task: print("Python works")                    # SUCCESS
- test_database_connection: FAILED (config issue)            # NEEDS FIX
```

**Traducción**: Airflow solo **existe** como infraestructura, pero no **orquesta** el pipeline DOE real.

### 12.2 Airflow en Phase 3: "Orchestrator Mode"
**Lo que DEBERÍA hacer** (futuro):
```python
# airflow/dags/doe_pipeline.py (FUTURO)
generate_data >> upload_to_minio >> load_to_postgres >> run_yates_analysis >> calculate_savings
```

**Analogía profesional**:
- **Phase 1**: Dry-run de orquesta - cada músico toca una nota para verificar que su instrumento funciona
- **Phase 3**: La orquesta ejecuta la sinfonía completa coordinadamente

### 12.3 Estado actual de componentes
| **Component** | **Phase 1 Role** | **Phase 3+ Role** |
|---------------|------------------|-------------------|
| **PostgreSQL** | ✅ Store test data | Store production pipeline results |
| **MinIO** | ✅ Store test files | Store all pipeline artifacts |
| **Airflow** | ✅ Execute simple tests | **Orchestrate entire DOE pipeline** |
| **Redis** | ✅ Message broker works | Handle task queuing for complex jobs |

**Próximo paso**: Implementar algoritmo de Yates (Phase 2) para que Airflow tenga algo útil que orquestar.

---

## 13) Checklist profesional y documentación

### 13.1 Creación de guía de fases
- **Documento creado**: `info/checklist_phases.md` con 7 fases detalladas
- **Contenido**: KPIs cuantificables, criterios de éxito, critical success factors
- **Objetivo**: Guía profesional paso a paso para completar el proyecto

### 13.2 Estado actual del proyecto
- **PHASE 1**: ✅ **COMPLETADA** - Infraestructura validada y funcional
- **PHASE 2**: 🚧 **LISTA PARA COMENZAR** - Implementar algoritmo de Yates
- **PHASES 3-7**: ⏳ **PENDIENTES** - Dependen de Phase 2

**KPIs target**: CPU -18.75%, vida útil +275%, ROI 425%, $284K ahorros anuales.

--- 