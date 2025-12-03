from signature_training import TrainingConfig, train_signature_model


if __name__ == "__main__":
    config = TrainingConfig(
        epochs=180,                    # 180 épocas entrenamiento completo
        steps_per_epoch=240,           # 1600 muestras / 64 batch = 25 * 10 = 250 (ajustado)
        val_steps=60,                  # Validación
        batch_size=64,                 # Batch size estándar
        hard_negatives=4,              # 4 negativos duros por triplet
        margin=0.25,                   # ⭐ REDUCIDO de 0.5 → Margen realista
        learning_rate=0.01,            # Learning rate
        lr_decay_steps=450,            # Decay schedule
        lr_decay_rate=0.94,            # Decay rate
        embedding_dim=256,             # ⭐ AUMENTADO de 128 → Más dimensiones
        model_path="embedding_network_final.h5",
        results_path="lstm_stacked_final_results.json",
        cache_path="lstm_stacked_final_SUCCESS.npz",
        x_path="Task2_Preprocesado/X_features.npy",
        y_path="Task2_Preprocesado/Y_user.npy",
        mask_path="Task2_Preprocesado/M_mask.npy",
        seed=42,
    )

    train_signature_model(config)
