# cifar10-loader

This is a GPU-accelerated dataloader for CIFAR-10 which does ~50 epochs/second on an NVIDIA A100.
For comparison, the PyTorch default does ~1 epoch/second.

It supports the following data augmentations:
* Random horizontal flip
* Random translation
* Cutout

## Installation
Clone then install via:
```
pip install -e .
```
Or, just copy-paste [loader.py](https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py) into your notebook.

# Usage

```
from quick_cifar import CifarLoader
loader = CifarLoader('cifar10_data_path/', train=True, batch_size=500,
                     aug=dict(flip=True, translate=4, cutout=8))
epochs = 100
for _ in range(epochs):
    for inputs, labels in loader:
        ...
```

