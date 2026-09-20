# Predicción de Supervivencia — Titanic (Kaggle)

Proyecto de clasificación binaria para predecir la supervivencia de pasajeros del Titanic usando el dataset
[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) de Kaggle. La variable objetivo es
`Survived` sobre 891 pasajeros de entrenamiento y 418 de test.

El pipeline completo vive en `notebooks/`, produce artefactos intermedios en `data/processed/` y las
predicciones finales en `outputs/`.

## Estructura de carpetas

```
proyecto_titanic/
├── data/
│   ├── raw/            # train.csv, test.csv, gender_submission.csv (Kaggle)
│   └── processed/      # train_clean.csv, test_clean.csv generados por 02_limpieza.ipynb
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_limpieza.ipynb
│   └── 03_modelo.ipynb
├── outputs/             # Predicciones finales (.csv) para submission a Kaggle
├── src/
│   └── utils.py
├── requirements.txt
└── README.md
```

## Descarga de datos

Los datos no están incluidos en el repositorio. Se obtienen desde la página de la competencia en Kaggle
(`train.csv`, `test.csv`, `gender_submission.csv`) y se colocan en `data/raw/`.

`test.csv` no contiene la columna `Survived`, ya que es el set de submission de la competencia. El pipeline de
evaluación de este proyecto usa un split estratificado 80/20 interno sobre `train.csv` para medir métricas
supervisadas de forma honesta antes de generar la predicción final sobre `test.csv`.

## Instalación de dependencias

```bash
python -m venv .venv
.venv/Scripts/activate      # En Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

## Resumen del pipeline

- **01_exploracion**: dimensiones, tipos de dato, porcentaje de faltantes por columna (`Age` 20%, `Cabin` 77%,
  `Embarked` 0.2%), distribución de la variable objetivo (~38% de supervivencia), y análisis de supervivencia
  cruzada por sexo, clase social (`Pclass`), tamaño de familia y puerto de embarque.
- **02_limpieza**: imputación de `Age` por mediana agrupada según `Pclass` y `Sex`; imputación de `Embarked` por
  moda; transformación de `Cabin` en variable binaria `HasCabin`; extracción de título honorífico (`Title`) desde
  `Name`, agrupando categorías poco frecuentes; extracción del prefijo de ticket (`Ticket_item`); construcción de
  `FamilySize` e `IsAlone` a partir de `SibSp` y `Parch`; segmentación de `Age` y `Fare` en bins categóricos.
- **03_modelo**: codificación one-hot de variables categóricas, split estratificado 80/20, entrenamiento y
  comparación de `LogisticRegression`, `RandomForestClassifier` y `GradientBoostingClassifier` (scikit-learn),
  optimización de hiperparámetros mediante `GridSearchCV` con validación cruzada de 5 folds, y diagnóstico de
  overfitting mediante comparación sistemática de métricas train/test.

## Diagnóstico y mitigación de overfitting

Con hiperparámetros por defecto, Random Forest memorizaba el conjunto de entrenamiento:

| Modelo | Accuracy train (antes) | Accuracy train (después) | Gap train−test (antes) | Gap train−test (después) |
|---|---:|---:|---:|---:|
| Random Forest (sin límite → `max_depth=5`, `min_samples_leaf=4`, `min_samples_split=10`) | 0.962 | 0.857 | 0.197 | 0.080 |
| Logistic Regression
