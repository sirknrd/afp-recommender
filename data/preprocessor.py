"""
AFP Dataset Preprocessor - Optimized
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PCA
from sklearn.model_selection import train_test_split
import torch

class AFPDataPreprocessor:
    def __init__(self, n_components=32, test_size=0.2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.test_size = test_size
        self.fitted = False
    
    def fit_transform(self, data_path):
        """Load and preprocess AFP data"""
        # Load data (ajusta la ruta a tu dataset)
        df = pd.read_csv(data_path)
        
        # Features y labels
        X = df.drop('target', axis=1).values
        y = df['target'].values.reshape(-1, 1)
        
        # Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        # Scale
        X_temp_scaled = self.scaler.fit_transform(X_temp)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA
        X_temp_pca = self.pca.fit_transform(X_temp_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        self.fitted = True
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_temp_pca.reshape(-1, 20, X_temp_pca.shape[1])
        X_test = X_test_pca.reshape(-1, 20, X_test_pca.shape[1])
        
        return (torch.FloatTensor(X_train), torch.FloatTensor(y_temp),
                torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    def transform(self, X):
        """Transform new data"""
        if not self.fitted:
            raise ValueError("Must fit preprocessor first")
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca.reshape(-1, 20, X_pca.shape[1])
