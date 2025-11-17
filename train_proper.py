from signature_training import TrainingConfig, train_signature_model


if __name__ == "__main__":
    config = TrainingConfig(
        epochs=180,
        steps_per_epoch=240,
        val_steps=60,
        batch_size=64,
        hard_negatives=4,
        margin=0.5,
        learning_rate=0.01,
        lr_decay_steps=450,
        lr_decay_rate=0.94,
        model_path="embedding_network_final.h5",
        results_path="lstm_stacked_final_results.json",
        cache_path="lstm_stacked_final_SUCCESS.npz",
        x_path="Task2_Preprocesado/X_features.npy",
        y_path="Task2_Preprocesado/Y_user.npy",
        mask_path="Task2_Preprocesado/M_mask.npy",
        seed=42,
    )

    train_signature_model(config)
