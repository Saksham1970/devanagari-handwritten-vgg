import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def prepare_data(dataset_path, num_imgs=None):
    """Prepare data for modeling
    output: image and label array"""

    types = ["Train", "Test"]
    output = []

    for type_ in types:
        labels = []
        images = []

        path_ = os.path.join(dataset_path, type_)
        for character in os.listdir(path_):
            character_path = os.path.join(path_, character)
            for i, image in enumerate(os.listdir(character_path)[:num_imgs]):
                labels.append(i)
                images.append(
                    Image.open(os.path.join(character_path, image)).convert("L")
                )

        np_images = np.empty(len(images), dtype="object")
        np_images[:] = images

        output.append([np_images, np.array(labels)])

    test = output.pop(-1)
    X_val, X_test, y_val, y_test = train_test_split(
        *test, test_size=0.5, random_state=42, stratify=test[-1]
    )
    output.append([X_val, y_val])
    output.append([X_test, y_test])

    return output


def get_dataloaders(path, batch_size=64, num_workers=0):
    """Prepare Train & Test dataloaders
    Augment training data using:
        - cropping
        - shifting (vertical/horizental)
        - horizental flipping
        - rotation

    input: path to FER+ Folder
    output: (Dataloader, Dataloader Dataloader)"""

    (xtrain, ytrain), (xval, yval), (xtest, ytest) = prepare_data(path)

    transform = transform = transforms.Compose([transforms.ToTensor()])

    
    train = CustomDataset(xtrain, ytrain, transform)
    val = CustomDataset(xval, yval, transform)
    test = CustomDataset(xtest, ytest, transform)

    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    valloader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    testloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return trainloader, valloader, testloader
