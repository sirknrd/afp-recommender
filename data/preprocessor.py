import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # ← AQUÍ ESTÁ LA FIX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Crea datos simulados AFP si no existen"""
    logger.info("📊 Creando datos simulados AFP...")
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'edad': np.random.randint(18, 70, n_samples),
        'ingreso_mensual': np.random.lognormal(11, 0.4, n_samples),
        'años_cotizando': np.random.randint(1, 40, n_samples),
        'riesgo_perfil': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'fondo_actual': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'retorno_A': np.random.normal(0.07, 0.12, n_samples),
        'retorno_B': np.random.normal(0.09, 0.15, n_samples),
        'retorno_C': np.random.normal(0.11, 0.18, n_samples),
        'retorno_D': np.random.normal(0.14, 0.22, n_samples),
        'retorno_E': np.random.normal(0.17, 0.28, n_samples),
    })
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/fondos_afp.csv', index=False)
    logger.info(f"✅ {n_samples} registros creados")
    return df

def preprocess():
    """Preprocesamiento completo"""
    # Cargar/crear datos
    try:
        df = pd.read_csv('data/raw/fondos_afp.csv')
    except:
        df = create_sample_data()
    
    # Features
    features = ['edad', 'ingreso_mensual', 'años_cotizando', 'riesgo_perfil',
                'retorno_A', 'retorno_B', 'retorno_C', 'retorno_D', 'retorno_E']
    
    X = df[features].fillna(df[features].mean())
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_scaled)
    
    # Guardar
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_processed.npy', X_pca)
    np.save('data/processed/scaler.npy', scaler)
    np.save('data/processed/pca.npy', pca)
    
    logger.info(f"✅ Preprocesado: {X_pca.shape}")
    print("🎉 DATOS LISTOS PARA ENTRENAMIENTO")
    return X_pca

if __name__ == "__main__":
    preprocess()
