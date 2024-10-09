import pandas as pd
import re

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
import pandas as pd

