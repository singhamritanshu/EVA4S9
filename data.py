import torch 
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
import random
import numpy as np
from Albumentationtransform import AlbumentationTransforms
import albumentations as A



def load():
  #transform = transforms.Compose([
              #transforms.ToPILImage(),
              #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
              #transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
              #transforms.RandomCrop(32, padding=4),
              #transforms.Resize((300, 300)),
              #transforms.CenterCrop((100, 100)),
              #transforms.RandomHorizontalFlip(p=0.5),
              #transforms.RandomRotation(degrees=(-90, 90)),
              #transforms.RandomVerticalFlip(p=0.5),
              #transforms.ToTensor(),
              #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
          #])
  #test_transform = transforms.Compose([
            
              #transforms.ToTensor(),
             # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
          #])
  #b = AlbumentationTransforms()
  
  channel_means = (0.5, 0.5, 0.5)
  channel_stdevs = (0.5, 0.5, 0.5)
  train_transform = AlbumentationTransforms([
                                        A.Rotate((-30.0, 30.0)),
                                        A.HorizontalFlip(),
                                        A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                                        A.Normalize(mean=channel_means, std=channel_stdevs),
                                        A.Cutout(num_holes=1, max_h_size=16,max_w_size = 16,p=1) 
                                        ])
  # Test Phase transformations
  test_transform = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs)])


          # Training set and train loader
  train_set = torchvision.datasets.CIFAR10(root='./data', download = True, train = True, transform = train_transform)
  trainloader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True, num_workers=2)

          # Test set and test loader 
  test_set = torchvision.datasets.CIFAR10(root='./data', download = True, train = False, transform = test_transform)
  testloader = torch.utils.data.DataLoader(test_set, batch_size = 128, shuffle = True, num_workers = 2)

  print("Finished loading data")

  classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  print(classes)
  return classes, trainloader, testloader
