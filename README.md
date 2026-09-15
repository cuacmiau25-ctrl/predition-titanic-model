# 🚢 Titanic: Machine Learning from Disaster (Kaggle)

Este repositorio contiene mi solución al clásico reto de Kaggle **"Titanic: Machine Learning from Disaster"**, donde el objetivo es predecir qué pasajeros sobrevivieron al naufragio utilizando modelos de Machine Learning.

## 📊 Descripción del Proyecto
El dataset contiene información sobre los pasajeros (edad, sexo, clase de boleto, tarifa, etc.). A través de análisis exploratorio de datos (EDA), ingeniería de características y modelado predictivo, se construye un clasificador para predecir la variable objetivo `Survived`.

## 🛠️ Tecnologías y Librerías Utilizadas
* **Lenguaje:** Python 3.x
* **Análisis de datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression, XGBoost)

## 📈 Proceso de Desarrollo
1. **Análisis Exploratorio (EDA):** Identificación de valores nulos, distribución de variables y correlación con la supervivencia.
2. **Limpieza de Datos:** Imputación de edades faltantes y datos de embarque.
3. **Ingeniería de Características (Feature Engineering):** 
   * Extracción de títulos a partir del nombre (Mr, Miss, Mrs, etc.).
   * Creación de la variable tamaño de familia (`FamilySize`).
4. **Modelado y Evaluación:** Comparación de métricas como Accuracy y F1-Score entre diferentes algoritmos.

## 🏆 Resultados
* **Mejor Modelo:** Random Forest Classifier
* **Accuracy en Validación:** ~82%
* **Puntaje en Kaggle:** `[Tu puntaje aquí, ej. 0.7845]`


