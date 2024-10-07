import pandas as pd
import re

def load_data(file_path):
    """Carga un archivo CSV en un DataFrame y lo devuelve."""
    return pd.read_csv(file_path, index_col=0)

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

def count_investors(row):
    """Cuenta el número de inversores que no son 'No Investor'."""
    return sum(row[['Investor 1', 'Investor 2', 'Investor 3']].apply(lambda x: x != 'No Investor' and pd.notnull(x)))

def add_investor_count(df):
    """Agrega una columna que cuenta el número de inversores por compañía."""
    df['Number of Investors'] = df.apply(count_investors, axis=1)
    return df

def save_cleaned_data(df, file_path):
    """Guarda el DataFrame limpio en un archivo CSV."""
    df.to_csv(file_path, index=False)

def basic_stats(df):
    """Muestra estadísticas básicas sobre el DataFrame."""
    print("Información general sobre el DataFrame:")
    df.info()
    print("\nPrimeras 50 filas del DataFrame limpio:")
    print(df)

def valuation_stats(df):
    """Muestra estadísticas sobre la valoración."""
    valuation_stats = df['Valuation ($B)'].describe()
    print("\nEstadísticas de Valuation ($B):")
    print(valuation_stats)

def yearly_joined_stats(df):
    """Extrae y muestra estadísticas de uniones por año."""
    df['Year Joined'] = df['Date Joined'].dt.year
    year_joined_stats = df['Year Joined'].value_counts().sort_index()
    print("\nUniones por año:")
    print(year_joined_stats)

def frequency_stats(df):
    """Muestra la frecuencia de empresas por país y ciudad."""
    country_stats = df['Country'].value_counts()
    city_stats = df['City'].value_counts()
    print("\nFrecuencia de empresas por país:")
    print(country_stats)
    print("\nFrecuencia de empresas por ciudad:")
    print(city_stats)

def investor_stats(df):
    """Muestra estadísticas sobre los inversores."""
    investor1_stats = df['Investor 1'].value_counts()
    investor2_stats = df['Investor 2'].value_counts()
    investor3_stats = df['Investor 3'].value_counts()
    print("\nFrecuencia de Investor 1:")
    print(investor1_stats)
    print("\nFrecuencia de Investor 2:")
    print(investor2_stats)
    print("\nFrecuencia de Investor 3:")
    print(investor3_stats)
