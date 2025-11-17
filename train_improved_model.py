from signature_training import TrainingConfig, train_signature_model


if __name__ == "__main__":
    config = TrainingConfig(
        epochs=150,
        steps_per_epoch=220,
        val_steps=50,
        batch_size=56,
        hard_negatives=3,
        margin=0.5,
        learning_rate=0.008,
        lr_decay_steps=350,
        lr_decay_rate=0.95,
        model_path="embedding_network_final.h5",
        results_path="lstm_stacked_final_results.json",
        cache_path="lstm_stacked_final_SUCCESS.npz",
        seed=21,
    )

    train_signature_model(config)
