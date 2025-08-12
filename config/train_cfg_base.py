class base_config:

    # Training Config
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    MODEL_SAVE_DIR = "checkpoints"

    # Ds specific config
    NUM_CLASSES = 3
    NUM_WORKERS = 4

    # WANDB
    WANDB_PROJECT_NAME = "sample_project"
    WANDB_RUN_NAME = "sample_run"
