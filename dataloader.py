import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataloader(batch_size):
  root_of_data="/content/drive/My Drive/celeba"
  transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
  ds=torchvision.datasets.MNIST(root=root_of_data, train=True,transform=transform, download=True)
  dloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
  return dloader
