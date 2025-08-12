import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


from dataset_loader.plant_disease import plant_disease_ds
from config.resnet_train_cfg import resnet_train_cfg

def get_dataloaders(cfg: resnet_train_cfg) -> dict:

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Validation / inference (center crop + normalization)
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


    train_ds = plant_disease_ds(
        data_root=config.DATA_ROOT,
        subset="train",
        transforms=train_tfms)
    
    train_ds = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, 
                          shuffle=True, 
                          num_workers=cfg.NUM_WORKERS,
                          pin_memory=True,
                          prefetch_factor=4 if cfg.NUM_WORKERS>0 else None, # only used when num_workers > 0
                          persistent_workers=True,    # keep workers alive across epochs
                          )
    
    val_ds = plant_disease_ds(
        data_root=config.DATA_ROOT,
        subset="validation",
        transforms=val_tfms)
    val_ds = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=True,
                        prefetch_factor=4 if cfg.NUM_WORKERS>0 else None, # only used when num_workers > 0
                        persistent_workers=True,    # keep workers alive across epochs
                        )
    
    test_ds = plant_disease_ds(
        data_root=config.DATA_ROOT,
        subset="test",
        transforms=val_tfms)

    test_ds = DataLoader(test_ds,
                        batch_size=cfg.BATCH_SIZE, 
                        shuffle=False, 
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=True,
                        prefetch_factor=4 if cfg.NUM_WORKERS>0 else None,  # only used when num_workers > 0
                        persistent_workers=True,    # keep workers alive across epochs
                        )

    return {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    }



def get_resnet_model():

    from torchvision.models import resnet18, ResNet18_Weights

    ## define model
    weights = ResNet18_Weights.IMAGENET1K_V1  # pick the weights you use
    model = resnet18(weights=weights)

    # freeze backbone (layers will not be trained)
    # transfer learning
    for p in model.parameters():
        p.requires_grad = False

    # Replace the classification head
    in_features = model.fc.in_features  # 2048 for resnet50

    # fully connected (fc) layers
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),  # dropout prob
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, config.NUM_CLASSES)
    )

    for name, param in model.fc.named_parameters():
        print(name, param.requires_grad)


    return model


def do_one_epoch(model, ds, optimizer=None, is_train=True, print_batch_loss=False):

    if is_train: 
        assert optimizer is not None

    overall_loss = 0
    for bidx, batch in enumerate(ds):

        img = batch["img"].to(device=device, non_blocking=True)
        label = batch["id_cls"].to(device=device, non_blocking=True)
        
        if not is_train:
            with torch.inference_mode():
                outputs = model(img)
        else:
            outputs = model(img)

        loss = loss_ce(outputs, label)

        overall_loss += loss.item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_batch_loss:
            print(f" batch: [{bidx}/{len(ds)}] loss: {loss.item():.4f}")

    return overall_loss/len(ds)


if __name__=="__main__":

    # setup a config
    config = resnet_train_cfg()

    config.BATCH_SIZE = 128
    config.DATA_ROOT = os.path.join("data", "plant_disease_recognition")
    config.EPOCHS = 20
    config.LEARNING_RATE = 2e-3
    config.NUM_CLASSES = 3
    config.NUM_WORKERS = min(8, max(1, os.cpu_count() // 2))
    config.MODEL_SAVE_DIR = "checkpoints"
    config.WANDB_PROJECT_NAME = "fs_computer_vision"
    config.WANDB_RUN_NAME = "resnet18_plant_disease"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loaders = get_dataloaders(config)

    train_ds = data_loaders["train"]
    val_ds = data_loaders["validation"]
    test_ds = data_loaders["test"]


    ## get model
    model = get_resnet_model()
    model = model.to(device)

    # ## set optimizer and
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)

    # ## setup loss function
    loss_ce = torch.nn.CrossEntropyLoss().cuda()

    wandb_run = wandb.init(
        project=config.WANDB_PROJECT_NAME,
        name=config.WANDB_RUN_NAME,
        config=vars(config)
    )

    config.MODEL_SAVE_DIR = os.path.join(config.MODEL_SAVE_DIR,
                                         "plantdisease",
                                         "resnet",
                                         wandb_run.id)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    

    best_val_loss = np.inf
    for epoch in tqdm(range(config.EPOCHS)):

        ## train
        model.train()
        overall_train_loss = do_one_epoch(model=model, ds=train_ds, optimizer=optimizer, is_train=True)

        ## val
        model.eval()
        overall_val_loss = do_one_epoch(model=model,ds=val_ds, optimizer=None, is_train=False)

        print(f"train loss: {overall_train_loss:.4f} val loss: {overall_val_loss:.4f}")

        if overall_val_loss<best_val_loss:
            best_val_loss = overall_val_loss

            torch.save(model.state_dict(), 
                os.path.join(config.MODEL_SAVE_DIR, f"best_resnet_model.pth")
            )
        
        wandb.log({
            "train/loss": overall_train_loss,
            "val/loss": overall_val_loss 
        }, step=epoch)

    
    
    ## test

    # Load best model
    #           256         64
    # runid = ["eb9kemz6","kyx3m6va"][1]
    # config.MODEL_SAVE_DIR = os.path.join(config.MODEL_SAVE_DIR,
    #                                      "plantdisease",
    #                                      "resnet",
    #                                      wandb_run.id)

    loaded_dict = torch.load(os.path.join(config.MODEL_SAVE_DIR, f"best_resnet_model.pth"),
                             weights_only=True)
    model.load_state_dict(loaded_dict)
    model.eval()

    total_correct = 0
    total = 0
    for tdata in test_ds:

        img = tdata["img"].to(device=device, non_blocking=True)
        label = tdata["id_cls"].to(device=device, non_blocking=True)

        with torch.inference_mode():

            model_out = model(img) # batch, cls
            # (1, (44534, 323, 422424))
            softmaxLogits = torch.nn.functional.softmax(model_out, dim=-1)
            # (1, (0.2, 0.5, 0.3))
            preds = softmaxLogits.argmax(dim=-1)

            correct = preds[preds==label].sum()

            total += label.size(0)
            total_correct += correct.item()
    
    print(f"Test Accuracy: {total_correct/total:.4f} for run: {wandb_run.id}")