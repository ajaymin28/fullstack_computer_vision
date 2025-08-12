from config.train_cfg_base import base_config

class resnet_train_cfg(base_config):

    BATCH_SIZE = 8
    EPOCHS = 10
    DATA_ROOT = ""
    NUM_CLASSES = 3
