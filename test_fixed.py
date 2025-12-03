#!/usr/bin/env python3
"""
Mini-entrenamiento de validación: 10% datos, parámetros corregidos
CORRECCIÓN: Padea datos de 208x4 a 400x4 para que coincida con máscara
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.spatial.distance import pdist, squareform

print("=" * 80)
print("MINI-ENTRENAMIENTO: Validación de parámetros corregidos")
print("=" * 80)

# Configuración
EPOCHS = 5
BATCH_SIZE = 16
MARGIN = 0.25
EMBEDDING_DIM = 256
LEARNING_RATE = 0.01

print(f"\n[CONFIG]")
print(f"  Epochs: {EPOCHS}")
print(f"  Margin: {MARGIN} (corrected from 0.5)")
print(f"  Embedding Dim: {EMBEDDING_DIM} (corrected from 128)")
print(f"  L2-Norm: NO (eliminada)")

print("\n[1/5] Cargando datos...")
X = np.load("Task2_Preprocesado/X_features.npy").astype(np.float32)  # (1600, 208, 4)
Y = np.load("Task2_Preprocesado/Y_user.npy").astype(np.int32)        # (1600,)
M = np.load("Task2_Preprocesado/M_mask.npy").astype(np.float32)      # (1600, 400)

print(f"  Original: X={X.shape}, Y={Y.shape}, M={M.shape}")

# PADDING: Extender de 208 a 400 timesteps con zeros
X_padded = np.zeros((len(X), 400, 4), dtype=np.float32)
X_padded[:, :X.shape[1], :] = X  # Copiar primeros 208 timesteps
X = X_padded

# Seleccionar 10% para mini-test
n_samples = int(len(X) * 0.10)
indices = np.random.choice(len(X), n_samples, replace=False)
X = X[indices]
Y = Y[indices]
M = M[indices]

print(f"  Mini (10%): X={X.shape}, Y={Y.shape}, M={M.shape}")

# Split 64/16/20
n_train = int(0.64 * len(X))
n_val = int(0.16 * len(X))

X_train, Y_train = X[:n_train], Y[:n_train]
X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

print("\n[2/5] Construyendo modelo...")

# Modelo LSTM sin L2-normalize
inputs = keras.Input(shape=(400, 4), name="signature")
mask_layer = layers.Masking(mask_value=0.0)(inputs)

lstm1 = layers.LSTM(128, return_sequences=True)(mask_layer)
lstm2 = layers.LSTM(64, return_sequences=False)(lstm1)

dense1 = layers.Dense(256, activation="relu")(lstm2)
embedding = layers.Dense(EMBEDDING_DIM, activation=None)(dense1)
# SIN L2-normalization (eliminada)

model = keras.Model(inputs=inputs, outputs=embedding, name="lstm_encoder")
print(f"  Parámetros: {model.count_params():,}")
print(f"  Embedding dim: {EMBEDDING_DIM}")
print(f"  Margen: {MARGIN}")
print(f"  L2-Norm: NO")

# Triplet loss
def triplet_loss(anchor, positive, negative, margin):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

print("\n[3/5] Entrenando...")

for epoch in range(EPOCHS):
    train_loss = 0
    
    for step in range(10):
        # Batch aleatorio
        batch_idx = np.random.choice(len(X_train), BATCH_SIZE, replace=True)
        anchor_data = X_train[batch_idx]
        anchor_y = Y_train[batch_idx]
        
        # Positivos (mismo usuario)
        positive_data = []
        for i, y in enumerate(anchor_y):
            same_user = np.where(Y_train == y)[0]
            same_user = same_user[same_user != batch_idx[i]]
            if len(same_user) > 0:
                positive_data.append(X_train[np.random.choice(same_user)])
            else:
                positive_data.append(anchor_data[i])
        positive_data = np.array(positive_data)
        
        # Negativos (diferente usuario)
        negative_data = []
        for y in anchor_y:
            diff_user = np.where(Y_train != y)[0]
            negative_data.append(X_train[np.random.choice(diff_user)])
        negative_data = np.array(negative_data)
        
        with tf.GradientTape() as tape:
            emb_a = model(anchor_data, training=True)
            emb_p = model(positive_data, training=True)
            emb_n = model(negative_data, training=True)
            loss = triplet_loss(emb_a, emb_p, emb_n, MARGIN)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_loss += float(loss)
    
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} - loss: {train_loss/10:.5f}")

print("\n[4/5] Evaluando...")

# Predicciones
train_emb = model.predict(X_train, verbose=0)
val_emb = model.predict(X_val, verbose=0)
test_emb = model.predict(X_test, verbose=0)

# Función EER
def compute_eer(embeddings, labels):
    distances = squareform(pdist(embeddings, metric='euclidean'))
    
    genuine_distances = []
    impostor_distances = []
    
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            if labels[i] == labels[j]:
                genuine_distances.append(distances[i, j])
            else:
                impostor_distances.append(distances[i, j])
    
    genuine_distances = np.array(genuine_distances)
    impostor_distances = np.array(impostor_distances)
    
    thresholds = np.linspace(0, max(genuine_distances.max(), impostor_distances.max()), 100)
    min_eer = 1.0
    
    for t in thresholds:
        far = np.sum(impostor_distances < t) / len(impostor_distances) if len(impostor_distances) > 0 else 0
        frr = np.sum(genuine_distances >= t) / len(genuine_distances) if len(genuine_distances) > 0 else 0
        eer = (far + frr) / 2
        if eer < min_eer:
            min_eer = eer
    
    return {
        "eer": min_eer,
        "genuine_mean": float(np.mean(genuine_distances)) if len(genuine_distances) > 0 else 0,
        "genuine_std": float(np.std(genuine_distances)) if len(genuine_distances) > 0 else 0,
        "impostor_mean": float(np.mean(impostor_distances)) if len(impostor_distances) > 0 else 0,
        "impostor_std": float(np.std(impostor_distances)) if len(impostor_distances) > 0 else 0,
    }

val_metrics = compute_eer(val_emb, Y_val)
test_metrics = compute_eer(test_emb, Y_test)

print("\n" + "=" * 80)
print("RESULTADOS")
print("=" * 80)

print(f"\nVALIDACION:")
print(f"  EER: {val_metrics['eer']*100:6.2f}%")
print(f"  Genuine: μ={val_metrics['genuine_mean']:.6f}, σ={val_metrics['genuine_std']:.6f}")
print(f"  Impostor: μ={val_metrics['impostor_mean']:.6f}, σ={val_metrics['impostor_std']:.6f}")

print(f"\nTEST:")
print(f"  EER: {test_metrics['eer']*100:6.2f}%")
print(f"  Genuine: μ={test_metrics['genuine_mean']:.6f}, σ={test_metrics['genuine_std']:.6f}")
print(f"  Impostor: μ={test_metrics['impostor_mean']:.6f}, σ={test_metrics['impostor_std']:.6f}")

print(f"\nEMBEDDING STATS:")
print(f"  Range: [{test_emb.min():.6f}, {test_emb.max():.6f}]")
print(f"  Mean: {test_emb.mean():.6f}, Std: {test_emb.std():.6f}")

# Guardar
results = {
    "config": {
        "epochs": EPOCHS,
        "margin": MARGIN,
        "embedding_dim": EMBEDDING_DIM,
        "l2_normalization": False,
    },
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "embedding_stats": {
        "min": float(test_emb.min()),
        "max": float(test_emb.max()),
        "mean": float(test_emb.mean()),
        "std": float(test_emb.std()),
    }
}

with open("test_fixed_results.json", "w") as f:
    json.dump(results, f, indent=2)

model.save("embedding_network_test_fixed.h5")

print("\n" + "=" * 80)
print("ARCHIVOS GUARDADOS")
print("  - test_fixed_results.json")
print("  - embedding_network_test_fixed.h5")
print("=" * 80)
