"""
Spanish to English translation dictionary for learning content.
Maintains consistency across all modules.
"""

import re

SPANISH_ENGLISH_DICT = {
    # Module/section names
    "Módulo": "Module",
    "módulo": "module",
    "Módulos": "Modules",
    "módulos": "modules",
    "Lección": "Lesson",
    "lección": "lesson",
    "Lecciones": "Lessons",
    "lecciones": "lessons",
    
    # Content sections
    "Introducción": "Introduction",
    "introducción": "introduction",
    "Descripción": "Description",
    "descripción": "description",
    "Objetivo": "Objective",
    "objetivo": "objective",
    "Objetivos": "Objectives",
    "objetivos": "objectives",
    "Contenido": "Content",
    "contenido": "content",
    "Contenidos": "Contents",
    "contenidos": "contents",
    
    # Learning materials
    "Práctica": "Practice",
    "práctica": "practice",
    "Prácticas": "Practices",
    "prácticas": "practices",
    "Ejercicio": "Exercise",
    "ejercicio": "exercise",
    "Ejercicios": "Exercises",
    "ejercicios": "exercises",
    "Ejemplo": "Example",
    "ejemplo": "example",
    "Ejemplos": "Examples",
    "ejemplos": "examples",
    "Problema": "Problem",
    "problema": "problem",
    "Problemas": "Problems",
    "problemas": "problems",
    
    # Learning content
    "Teoría": "Theory",
    "teoría": "theory",
    "Evaluación": "Evaluation",
    "evaluación": "evaluation",
    "Mini-proyecto": "Mini-project",
    "mini-proyecto": "mini-project",
    "Notas": "Notes",
    "notas": "notes",
    "Instrucciones": "Instructions",
    "instrucciones": "instructions",
    "Sección": "Section",
    "sección": "section",
    "Tema": "Topic",
    "tema": "topic",
    "Temas": "Topics",
    "temas": "topics",
    "Concepto": "Concept",
    "concepto": "concept",
    "Conceptos": "Concepts",
    "conceptos": "concepts",
    
    # Programming concepts
    "Aprendizaje": "Learning",
    "aprendizaje": "learning",
    "Algoritmo": "Algorithm",
    "algoritmo": "algorithm",
    "Algoritmos": "Algorithms",
    "algoritmos": "algorithms",
    "Función": "Function",
    "función": "function",
    "Funciones": "Functions",
    "funciones": "functions",
    "Variable": "Variable",
    "variable": "variable",
    "Variables": "Variables",
    "variables": "variables",
    "Datos": "Data",
    "datos": "data",
    "Tipo": "Type",
    "tipo": "type",
    "Tipos": "Types",
    "tipos": "types",
    "Estructura": "Structure",
    "estructura": "structure",
    "Estructuras": "Structures",
    "estructuras": "structures",
    
    # ML-specific
    "Modelo": "Model",
    "modelo": "model",
    "Modelos": "Models",
    "modelos": "models",
    "Entrenamiento": "Training",
    "entrenamiento": "training",
    "Prueba": "Testing",
    "prueba": "testing",
    "Validación": "Validation",
    "validación": "validation",
    "Predicción": "Prediction",
    "predicción": "prediction",
    "Predicciones": "Predictions",
    "predicciones": "predictions",
    "Métrica": "Metric",
    "métrica": "metric",
    "Métricas": "Metrics",
    "métricas": "metrics",
    "Hiperparámetro": "Hyperparameter",
    "hiperparámetro": "hyperparameter",
    "Hiperparámetros": "Hyperparameters",
    "hiperparámetros": "hyperparameters",
    "Regularización": "Regularization",
    "regularización": "regularization",
    "Overfitting": "Overfitting",
    "overfitting": "overfitting",
    "Underfitting": "Underfitting",
    "underfitting": "underfitting",
    
    # Data science
    "Conjunto de datos": "Dataset",
    "conjunto de datos": "dataset",
    "Conjuntos de datos": "Datasets",
    "conjuntos de datos": "datasets",
    "Exploración": "Exploration",
    "exploración": "exploration",
    "Limpieza": "Cleaning",
    "limpieza": "cleaning",
    "Normalización": "Normalization",
    "normalización": "normalization",
    "Características": "Features",
    "características": "features",
    "Característica": "Feature",
    "característica": "feature",
    "Valor faltante": "Missing value",
    "valores faltantes": "missing values",
    "Outlier": "Outlier",
    "outlier": "outlier",
    "Outliers": "Outliers",
    "outliers": "outliers",
    "Visualización": "Visualization",
    "visualización": "visualization",
    
    # Deep Learning
    "Red neuronal": "Neural network",
    "redes neuronales": "neural networks",
    "Neurona": "Neuron",
    "neurona": "neuron",
    "Neuronas": "Neurons",
    "neuronas": "neurons",
    "Capa": "Layer",
    "capa": "layer",
    "Capas": "Layers",
    "capas": "layers",
    "Activación": "Activation",
    "activación": "activation",
    "Propagación": "Propagation",
    "propagación": "propagation",
    "Gradient": "Gradient",
    "gradient": "gradient",
    "Descenso de gradiente": "Gradient descent",
    "descenso de gradiente": "gradient descent",
    
    # NLP
    "Procesamiento de texto": "Text processing",
    "procesamiento de texto": "text processing",
    "Token": "Token",
    "token": "token",
    "Tokens": "Tokens",
    "tokens": "tokens",
    "Tokenización": "Tokenization",
    "tokenización": "tokenization",
    "Embedding": "Embedding",
    "embedding": "embedding",
    "Embeddings": "Embeddings",
    "embeddings": "embeddings",
    "Sentimiento": "Sentiment",
    "sentimiento": "sentiment",
    "Clasificación": "Classification",
    "clasificación": "classification",
    "Clasificador": "Classifier",
    "clasificador": "classifier",
    
    # Computer Vision
    "Imagen": "Image",
    "imagen": "image",
    "Imágenes": "Images",
    "imágenes": "images",
    "Canal": "Channel",
    "canal": "channel",
    "Canales": "Channels",
    "canales": "channels",
    "Pixel": "Pixel",
    "pixel": "pixel",
    "Píxeles": "Pixels",
    "píxeles": "pixels",
    "Convolución": "Convolution",
    "convolución": "convolution",
    "Filtro": "Filter",
    "filtro": "filter",
    "Filtros": "Filters",
    "filtros": "filters",
    
    # Time Series
    "Serie temporal": "Time series",
    "series temporales": "time series",
    "Pronóstico": "Forecast",
    "pronóstico": "forecast",
    "Pronósticos": "Forecasts",
    "pronósticos": "forecasts",
    "Predicción": "Prediction",
    "predicción": "prediction",
    "Tendencia": "Trend",
    "tendencia": "trend",
    "Estacionalidad": "Seasonality",
    "estacionalidad": "seasonality",
    "Anomalía": "Anomaly",
    "anomalía": "anomaly",
    "Anomalías": "Anomalies",
    "anomalías": "anomalies",
    
    # Evaluation
    "Precisión": "Precision",
    "precisión": "precision",
    "Recall": "Recall",
    "recall": "recall",
    "F1": "F1",
    "f1": "f1",
    "Accuracy": "Accuracy",
    "accuracy": "accuracy",
    "Matriz de confusión": "Confusion matrix",
    "matriz de confusión": "confusion matrix",
    "Curva ROC": "ROC curve",
    "curva ROC": "ROC curve",
    "AUC": "AUC",
    "auc": "auc",
    "Error": "Error",
    "error": "error",
    "Errores": "Errors",
    "errores": "errors",
    
    # General
    "Introducción": "Introduction",
    "introducción": "introduction",
    "Conclusión": "Conclusion",
    "conclusión": "conclusion",
    "Conclusiones": "Conclusions",
    "conclusiones": "conclusions",
    "Recursos": "Resources",
    "recursos": "resources",
    "Referencias": "References",
    "referencias": "references",
    "Bibliografía": "Bibliography",
    "bibliografía": "bibliography",
    "Requisitos": "Requirements",
    "requisitos": "requirements",
    "Instalación": "Installation",
    "instalación": "installation",
    "Uso": "Usage",
    "uso": "usage",
    "Ejemplo de uso": "Usage example",
    "ejemplo de uso": "usage example",
    "Resultado": "Result",
    "resultado": "result",
    "Resultados": "Results",
    "resultados": "results",
    "Análisis": "Analysis",
    "análisis": "analysis",
    "Conclusión": "Conclusion",
    "conclusión": "conclusion",
}

# File name mappings
FILE_NAME_MAPPINGS = {
    # General
    "vectores-producto-punto": "vectors-dot-product",
    "estadistica-descriptiva": "descriptive-statistics",
    "escala-caracteristicas": "feature-scaling",
    "regresion-lineal": "linear-regression",
    "gradiente-un-parametro": "gradient-descent-one-parameter",
    "pipeline-matematico": "mathematical-pipeline",
    
    # Data
    "limpieza-dataset": "dataset-cleaning",
    "eda-completo": "complete-eda",
    "valores-faltantes": "missing-values",
    "imputacion": "imputation",
    "deteccion-outliers": "outlier-detection",
    "calidad-datos": "data-quality",
    "pipeline-limpieza": "cleaning-pipeline",
    
    # ML
    "pipeline-clasificacion": "classification-pipeline",
    "regresion-diagnostico": "regression-diagnostics",
    "train-test": "train-test-split",
    "lineal-vs-logistica": "linear-vs-logistic",
    "regularizacion": "regularization",
    "profundidad-overfitting": "tree-depth-overfitting",
    "comparacion-modelos": "model-comparison",
    
    # DL
    "red-neuronal-simple": "simple-neural-network",
    "red-neuronal": "neural-network",
    "dropout": "dropout",
    "neurona-forward": "neuron-forward",
    "descenso-gradiente": "gradient-descent",
    "backpropagation": "backpropagation",
    
    # NLP
    "clasificacion-sentimientos": "sentiment-classification",
    "prompt-engineering": "prompt-engineering",
    "procesamiento-texto": "text-processing",
    "embeddings": "embeddings",
    "similitud-coseno": "cosine-similarity",
    "rag-evaluacion": "rag-evaluation",
    
    # CV
    "clasificacion-imagenes": "image-classification",
    "transfer-learning": "transfer-learning",
    "deteccion-objetos": "object-detection",
    "segmentacion": "segmentation",
    "aumentacion": "augmentation",
    
    # Time Series
    "pronostico-series": "time-series-forecasting",
    "deteccion-anomalias": "anomaly-detection",
    "backtesting": "backtesting",
    
    # Recommender
    "collaborative-filtering": "collaborative-filtering",
    "matrix-factorization": "matrix-factorization",
    "content-based": "content-based",
    "cold-start": "cold-start",
    
    # Generative AI
    "rag-retrieval": "rag-retrieval",
    "gan-generacion": "gan-generation",
    "vae-diffusion": "vae-diffusion",
    
    # Ethics
    "deteccion-bias": "bias-detection",
    "fairness": "fairness",
    "explainability": "explainability",
    "shap-lime": "shap-lime",
    
    # Security
    "anonimizacion": "anonymization",
    "k-anonymity": "k-anonymity",
    "differential-privacy": "differential-privacy",
    "encriptacion": "encryption",
    "hashing": "hashing",
    "federated": "federated",
    
    # MLOps
    "deployment": "deployment",
    "monitoring": "monitoring",
    "drift": "drift",
    "rollout": "rollout",
    "rollback": "rollback",
    "reproducibilidad": "reproducibility",
    "versionado": "versioning",
    "cicd": "cicd",
    "observability": "observability",
    
    # Practices
    "ingesta-perfilado": "ingestion-profiling",
    "limpieza-nulos": "null-cleaning",
    "normalizacion-tipos": "normalization-types",
    "outliers-calidad": "outliers-quality",
    "visualizacion-eda": "visualization-eda",
    "exploracion-temporal": "temporal-exploration",
    "baseline-regresion": "regression-baseline",
    "baseline-clasificacion": "classification-baseline",
    "baseline-forecasting": "forecasting-baseline",
    "validacion-cruzada": "cross-validation",
    "metricas-errores": "metrics-errors",
    "modelo-final": "final-model",
    "riesgos": "risks",
    "deteccion-sesgos": "bias-detection",
    "explicabilidad": "explainability",
    "mitigacion": "mitigation",
    "mapeo-riesgos": "risk-mapping",
    "plan-mitigacion": "mitigation-plan",
    "controles-acceso": "access-controls",
    "clasificacion-datos": "data-classification",
    "plan-respuesta": "response-plan",
    "incidentes": "incidents",
    "versionado-reproducibilidad": "versioning-reproducibility",
    "serving-basico": "basic-serving",
    "monitoreo": "monitoring",
    "monitoreo-modelo": "model-monitoring",
    "alertas": "alerts",
}

def translate_text(text: str) -> str:
    """
    Translate Spanish text to English using the dictionary.
    Handles case preservation and word boundaries.
    """
    result = text
    
    # Sort by length (longest first) to handle compound phrases
    sorted_pairs = sorted(SPANISH_ENGLISH_DICT.items(), key=lambda x: len(x[0]), reverse=True)
    
    for spanish, english in sorted_pairs:
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(spanish) + r'\b'
        result = re.sub(pattern, english, result, flags=re.IGNORECASE)
    
    return result


def translate_filename(filename: str) -> str:
    """Translate filename from Spanish to English."""
    name = filename
    
    # Handle .md and .py extensions
    ext = ""
    if name.endswith(".md"):
        ext = ".md"
        name = name[:-3]
    elif name.endswith(".py"):
        ext = ".py"
        name = name[:-3]
    
    # Apply mappings
    for spanish, english in FILE_NAME_MAPPINGS.items():
        name = name.replace(spanish, english)
    
    return name + ext


if __name__ == "__main__":
    import re
    # Test examples
    test_text = "Introducción a los algoritmos de aprendizaje automático"
    print(f"Original: {test_text}")
    print(f"Translated: {translate_text(test_text)}")
