
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def download_dataset(dataset_name, data_dir='./data'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')


def get_dataloaders(train_dataset, test_dataset, batch_size=64, val_split=0.2):
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def apply_augmentation(train_dataset, augmentations):
    train_dataset.transform = transforms.Compose(
        [augmentations, train_dataset.transform])
    return train_dataset
