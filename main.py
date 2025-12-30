from multiprocessing import freeze_support
from timed_decorator.simple_timed import timed

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


class CustomDataset(Dataset):
    def __init__(self, data_path: str = './data', train: bool = True, cache: bool = True):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


@timed(return_time=True, use_seconds=True, stdout=False)
def transform_dataset_with_transforms(dataset: TensorDataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])
    for image in dataset.tensors[0]:
        transforms(image)


@timed(return_time=True, use_seconds=True, stdout=False)
@torch.no_grad()
def transform_dataset_with_model(dataset: TensorDataset, model: nn.Module, batch_size: int, device: torch.device):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type != "cpu"),
        drop_last=False
    )
    for (images,) in dataloader:
        images = images.to(device, non_blocking=True)
        _ = model(images)


def test_inference_time(model: nn.Module, device=torch.device('cpu')):
    test_dataset = CustomDataset(train=False, cache=False)
    test_dataset = torch.stack(test_dataset.images)  # (N,3,32,32)
    test_dataset = TensorDataset(test_dataset)

    for batch_size in [1, 8, 16, 32, 64, 128, 256, 512]:
        _, t1 = transform_dataset_with_transforms(test_dataset)
        _, t2 = transform_dataset_with_model(test_dataset, model, batch_size, device)
        print(
            f"[{device.type}] batch_size={batch_size:4d} | "
            f"transforms CPU seq: {t1:.4f}s | model: {t2:.4f}s"
        )


def main():
    print("Ruleaza inference.py ")


if __name__ == '__main__':
    freeze_support()
    main()
