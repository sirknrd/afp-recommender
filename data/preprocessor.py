import pandas as pd
import numpy as np
import os

print("🚀 Iniciando preprocesamiento AFP...")

# Crear datos simulados AFP
np.random.seed(42)
n_samples = 5000

data = {
    'edad': np.random.randint(18, 70, n_samples),
    'ingreso': np.random.lognormal(11, 0.4, n_samples),
    'riesgo': np.random.choice([1,2,3,4,5], n_samples),
    'retorno_A': np.random.normal(0.07, 0.12, n_samples),
    'retorno_B': np.random.normal(0.09, 0.15, n_samples),
    'retorno_C': np.random.normal(0.11, 0.18, n_samples),
    'retorno_D': np.random.normal(0.14, 0.22, n_samples),
    'retorno_E': np.random.normal(0.17, 0.28, n_samples),
}

df = pd.DataFrame(data)

# Normalizar manualmente (sin sklearn)
for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()

# Guardar
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', df.values)
df.to_csv('data/processed/afp_data.csv', index=False)

print(f"✅ DATOS CREADOS: {df.shape}")
print("📁 Archivos:")
print("- data/processed/X_train.npy")
print("- data/processed/afp_data.csv")
print("🎉 ¡LISTO PARA ENTRENAR!")
