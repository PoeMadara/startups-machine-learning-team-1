import os  # Importa el módulo os para interactuar con el sistema operativo
import sys  # Importa el módulo sys para acceder a algunas variables y funciones del intérprete
import re  # Importa el módulo re para trabajar con expresiones regulares
import pandas as pd  # Importa pandas como pd para la manipulación y análisis de datos
import matplotlib.pyplot as plt  # Importa pyplot de matplotlib para la creación de gráficos
import seaborn as sns  # Importa seaborn como sns para visualizaciones estadísticas mejoradas
from sklearn.model_selection import train_test_split, GridSearchCV  # Importa funciones para dividir datos y realizar búsqueda de hiperparámetros
from sklearn.linear_model import LogisticRegression  # Importa la clase para crear un modelo de regresión logística
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report  # Importa métricas para evaluar modelos
from sklearn.tree import DecisionTreeClassifier  # Importa la clase para crear un modelo de árbol de decisión
from sklearn.svm import SVC  # Importa la clase para crear un clasificador de máquinas de soporte (SVM)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Importa clases para crear modelos de Random Forest y Gradient Boosting
from sklearn.preprocessing import StandardScaler  # Importa StandardScaler para escalar características
from imblearn.over_sampling import SMOTE  # Importa SMOTE para balancear clases en conjuntos de datos desiguales
from sklearn.exceptions import UndefinedMetricWarning  # Importa la advertencia para métricas indefinidas en sklearn



# ---------------CARGA DE DATOS-------------------
def load_data(file_path):
    """Carga un archivo CSV en un DataFrame y lo devuelve."""
    return pd.read_csv(file_path, index_col=0)

# -------------FUNCIONES DE LIMPIEZA---------------
def clean_text(text):
    """Limpia el texto mediante la conversión a minúsculas, eliminación de caracteres especiales y recorte de espacios."""
    if isinstance(text, str):
        text = text.lower()  # Convertir a minúsculas
        text = re.sub(r'[^\w\s&.,]', '', text)  # Eliminar caracteres especiales
        text = text.strip()  # Eliminar espacios extra
        text = text.title()  # Poner en mayúscula la primera letra de cada palabra
    return text

def clean_dataframe(df):
    """Aplica varias funciones de limpieza a un DataFrame."""
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Lista de columnas que se quieren limpiar
    columns_to_clean = ['Company', 'Country', 'City', 'Industry', 'Investor 1', 'Investor 2', 'Investor 3']
    for column in columns_to_clean:
        df[column] = df[column].apply(clean_text)

    df = df.drop('Investor 4', axis=1)

    df['Investor 2'] = df['Investor 2'].fillna('No Investor')
    df['Investor 3'] = df['Investor 3'].fillna('No Investor')

    df['Valuation ($B)'] = df['Valuation ($B)'].apply(lambda x: re.sub(r'[^\d.]', '', str(x)))
    df['Valuation ($B)'] = df['Valuation ($B)'].astype(float)

    df['Date Joined'] = pd.to_datetime(df['Date Joined'], errors='coerce')

    return df

# --------------FUNCIONES DE ANÁLISIS---------------
def count_investors(row): # Esta función se usa dentro de add_investor_count y no se llama directamente desde el main
    """Cuenta el número de inversores que no son 'No Investor'."""
    return sum(row[['Investor 1', 'Investor 2', 'Investor 3']].apply(lambda x: x != 'No Investor' and pd.notnull(x)))

def add_investor_count(df):
    """Agrega una columna que cuenta el número de inversores por compañía."""
    df['Number of Investors'] = df.apply(count_investors, axis=1)
    return df

# --------------MODELOS CON HIPERPARÁMETROS AJUSTADOS---------------
def train_logistic_regression_model(X_train, X_test, y_train, y_test):
    """Función para entrenar y evaluar un modelo de Regresión Logística.

    Args:
        X_train (np.ndarray): Conjunto de características de entrenamiento.
        X_test (np.ndarray): Conjunto de características de prueba.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        None
    """

    # ---------- Entrenar y evaluar el modelo (Logistic Regression) ----------
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)  # Usar los datos escalados
    y_pred = clf.predict(X_test)  # Usar los datos escalados para predicciones

    # Evaluación del modelo
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))  # Añadir zero_division=0

    # Matriz de confusión y reporte de clasificación
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))  # Añadir zero_division=0


def train_random_forest_model(X_train, X_test, y_train, y_test):
    """Función para entrenar y evaluar un modelo de Random Forest.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        None
    """
    
    # Verificar el tamaño de las clases en el conjunto de entrenamiento
    print("Distribución de clases en y_train antes de SMOTE:")
    print(y_train.value_counts())

    # ---------- Aplicar SMOTE para el balanceo de clases ----------
    smote = SMOTE(random_state=42, k_neighbors=1)  # No cambiar este valor
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Verificar el tamaño de las clases después de SMOTE
    print("Distribución de clases en y_train después de SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # ---------- Escalado de características (opcional, pero recomendado) ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Definir los hiperparámetros para probar ----------
    param_grid = {
        'n_estimators': [100, 200, 300],  # Número de árboles
        'max_depth': [10, 20, 30, None],  # Profundidad máxima
        'min_samples_split': [2, 5, 10],  # Tamaño mínimo para dividir nodos
        'min_samples_leaf': [1, 2, 4],  # Tamaño mínimo de las hojas
    }

    # Crear el modelo
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Configurar la búsqueda de cuadrícula
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Ajustar el modelo
    grid_search.fit(X_train_scaled, y_train_resampled)

    # Mejor resultado
    print(f"Mejor score: {grid_search.best_score_}")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")

    # Crear un nuevo clasificador con los mejores parámetros
    best_rf_clf = RandomForestClassifier(
        n_estimators=grid_search.best_params_['n_estimators'],
        max_depth=grid_search.best_params_['max_depth'],
        min_samples_split=grid_search.best_params_['min_samples_split'],
        min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
        random_state=42
    )

    # Entrenar el modelo en los datos de entrenamiento reequilibrados
    best_rf_clf.fit(X_train_resampled, y_train_resampled)

    # Hacer predicciones en el conjunto de prueba
    y_pred_best_rf = best_rf_clf.predict(X_test_scaled)

    # Evaluar el modelo
    print("Random Forest (mejores parámetros) - Accuracy:", accuracy_score(y_test, y_pred_best_rf))
    print("Random Forest (mejores parámetros) - F1 Score:", f1_score(y_test, y_pred_best_rf, average='weighted'))

    # Matriz de confusión y reporte de clasificación
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_best_rf))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_best_rf, zero_division=0))


def train_gradient_boosting_model(X_train, X_test, y_train, y_test):
    """Función para entrenar y evaluar un modelo de Gradient Boosting.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        None
    """

    # Verificar el tamaño de las clases en el conjunto de entrenamiento
    print("Distribución de clases en y_train antes de SMOTE:")
    print(y_train.value_counts())

    # ---------- Aplicar SMOTE para el balanceo de clases ----------
    smote = SMOTE(random_state=42, k_neighbors=1)  # Cambiar el número de vecinos a 1
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Verificar el tamaño de las clases después de SMOTE
    print("Distribución de clases en y_train después de SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # ---------- Escalado de características ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # ---------- Definir los hiperparámetros para probar ----------
    param_grid = {
        'n_estimators': [100, 150],  # Reducir el rango para acelerar el tiempo de entrenamiento
        'learning_rate': [0.01, 0.05, 0.1],  # Añadir learning_rate
        'max_depth': [3, 5],  # Reducir profundidad máxima
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.7, 0.8, 0.9]  # Añadir subsample
    }

    # Crear el modelo
    gb_clf = GradientBoostingClassifier(random_state=42)

    # Configurar la búsqueda de cuadrícula
    grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    # Ajustar el modelo
    grid_search.fit(X_train_scaled, y_train_resampled)

    # Mejor resultado
    print(f"Mejor score: {grid_search.best_score_}")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")

    # Evaluar el mejor modelo encontrado por la búsqueda de cuadrícula
    best_gb_clf = grid_search.best_estimator_
    y_pred_gb = best_gb_clf.predict(X_test_scaled)  # Hacer predicciones

    # Evaluación del modelo
    print("Gradient Boosting - Accuracy:", accuracy_score(y_test, y_pred_gb))
    print("Gradient Boosting - F1 Score:", f1_score(y_test, y_pred_gb, average='weighted'))

    # Matriz de confusión y reporte de clasificación
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_gb))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_gb))


# --------------COMPARACIÓN DE CLASIFICADORES---------------
def evaluate_models(X_train_scaled, X_train_resampled, X_test_scaled, X_test, y_train_resampled, y_test):
    """Función para evaluar múltiples modelos de clasificación y comparar sus resultados.

    Args:
        X_train_scaled (pd.DataFrame): Conjunto de características de entrenamiento (escaladas).
        X_train_resampled (pd.DataFrame): Conjunto de características de entrenamiento reequilibrado con SMOTE.
        X_test_scaled (pd.DataFrame): Conjunto de características de prueba (escaladas).
        X_test (pd.DataFrame): Conjunto de características de prueba sin escalar (para Decision Tree).
        y_train_resampled (pd.Series): Variable objetivo de entrenamiento reequilibrado con SMOTE.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        None
    """

    # Lista de modelos para probar
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Classifier': SVC()
    }

    # Almacenar resultados en un DataFrame
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score'])

    # Evaluar cada modelo
    for model_name, model in models.items():
        # Entrenar el modelo
        model.fit(X_train_scaled if model_name != 'Decision Tree' else X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled if model_name != 'Decision Tree' else X_test)

        # Evaluación del modelo
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Almacenar resultados utilizando loc
        results.loc[len(results)] = {'Model': model_name, 'Accuracy': accuracy, 'F1 Score': f1}

        print(f"\n{model_name} - Accuracy: {accuracy:.4f}")
        print(f"{model_name} - F1 Score: {f1:.4f}")

        # Matriz de confusión y reporte de clasificación
        print(f"\n{model_name} - Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\n{model_name} - Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

    # Visualizar resultados
    results.set_index('Model', inplace=True)
    results.plot(kind='bar', figsize=(10, 5), title='Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.show()

# --------------MAPA DE CORRELACIÓN---------------
def plot_correlation_matrix(df):
    """Genera un mapa de correlación a partir de un DataFrame, utilizando Label Encoding para columnas no numéricas."""
    plt.figure(figsize=(12, 10))  # Ajusta el tamaño de la figura

    # Hacer una copia del DataFrame
    df_copy = df.copy()

    # Codificar columnas no numéricas utilizando Label Encoding
    for column in df_copy.select_dtypes(include=['object']).columns:
        df_copy[column] = df_copy[column].astype('category').cat.codes

    # Calcular la matriz de correlación
    correlation_matrix = df_copy.corr()

    # Crear el heatmap con todas las columnas
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap de correlación entre todas las columnas')
    plt.show()

# --------------FUNCIONES DE GUARDADO---------------
def save_cleaned_data(df, file_path):
    """Guarda el DataFrame limpio en un archivo CSV."""
    df.to_csv(file_path, index=False)
    print('Archivo guardado')
