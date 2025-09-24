import  os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       train_transform: Compose = None,
                       test_transform: Compose = None,
                       batch_size: int = 32,
                       num_workers: int = NUM_WORKERS, ):
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    class_names = train_data.classes
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,)
    return train_loader, test_loader,class_names

import inspect
print(inspect.signature(create_dataloaders))