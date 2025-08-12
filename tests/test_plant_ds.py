from dataset_loader.plant_disease import plant_disease_ds
from torchvision.transforms import transforms as T
import os


if __name__=="__main__":

    sample_transforms = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])
    # sample_transforms = None

    train_ds = plant_disease_ds(
        data_root=os.path.join("data", "plant_disease_recognition"),
        subset="train",
        transforms=sample_transforms)

    # print(str(train_ds))

    # print("sample from idx 0 ")
    # data_sample = train_ds[0]
    # for key,val in data_sample.items():
    #     print(f"[key]: {key} [val]: {val}")

    print("sample from idx -1 ")
    data_sample = train_ds[-1]
    for key,val in data_sample.items():
        print(f"[key]: {key} [val]: {val}")
    
    print(f"Size of ten: {data_sample['img'].shape}")