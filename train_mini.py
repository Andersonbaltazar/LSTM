"""
TRAIN_MINI (Actualizado): Modo de entrenamiento rápido y real usando Triplet Loss
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from signature_training import TrainingConfig, train_signature_model

if __name__ == "__main__":
    print("=" * 80)
    print("   TRAIN_MINI: INICIANDO ENTRENAMIENTO BIOMÉTRICO RÁPIDO (TRIPLET LOSS)")
    print("=" * 80)
    
    # Configuramos los parámetros idénticos al entrenamiento rápido exitoso
    config = TrainingConfig(
        epochs=15,                     # 15 épocas para que sea "mini" y rápido en local
        steps_per_epoch=100,           # Cantidad moderada de pasos por ciclo
        val_steps=20,                  # Pasos de validación cortos
        batch_size=32,                 # Tamaño de lote eficiente para CPU/GPU local
        hard_negatives=2,              # Negativos duros para empezar a empujar vectores
        margin=0.25,                   # Margen real de separación de Triplet Loss
        learning_rate=0.001,           # Tasa de aprendizaje controlada
        embedding_dim=256,             # 256 dimensiones (la arquitectura real del sistema)
        model_path="embedding_network_mini.h5", # Guardamos sobre el nombre original del mini
        results_path="mini_triplet_results.json",
        cache_path="mini_triplet_SUCCESS.npz",
        x_path="Task2_Preprocesado/X_features.npy",
        y_path="Task2_Preprocesado/Y_user.npy",
        mask_path="Task2_Preprocesado/M_mask.npy",
        seed=42,
    )

    print("[INFO] Lanzando el motor de entrenamiento...")
    train_signature_model(config)
    print("\n[OK] ¡Proceso terminado con éxito!")
    print("El archivo 'embedding_network_mini.h5' ahora es seguro y usa Triplet Loss.")