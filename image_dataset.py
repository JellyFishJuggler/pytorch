import torch
import torch.nn as nn
import torch.optim as op
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pandas as py

IMG_PATH = './data/FETAL_PLANES_ZENODO/Images/Patient00001_Plane1_2_of_15.png' # Patient00001_Plane1_1_of_15.png
image = Image.open(IMG_PATH)

print(type(image))
print(image.size)
print(image.mode)

r, g, b, a = image.split()

print(r.size)
print(g.size)
print(b.size)
print(a.size)

# mp.imshow(image, cmap='gray')
# mp.axis('off')
# mp.show()
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(r, cmap="gray")
plt.title("Red")

plt.subplot(1, 3, 2)
plt.imshow(g, cmap="gray")
plt.title("Green")

plt.subplot(1, 3, 3)
plt.imshow(b, cmap="gray")
plt.title("Blue")

plt.show()