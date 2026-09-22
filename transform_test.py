# pyright: reportMissingModuleSource=false

import torch
from torchvision import transforms, datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as mp

# rotation = transforms.RandomRotation(10)

train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=test_transform
)

image,label = train_data[0]

# rotated_img = train_transform(image)


print(type(image))
# print(type(rotated_img))
print(image.size)
# print(rotated_img.size)


print(train_data[0][0].shape)
print(test_data[0][0].shape)

print(type(image))
print(image.min())
print(image.max())
print(image.shape)

# mp.figure(figsize=(8,8))
# mp.imshow(image, cmap='gray')
# mp.show()

# mp.imshow(rotated_img, cmap='gray')
# mp.show()