import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
# Importing Modules 
import data as d 
import show_images as s 
import model as m 
import train_test as t 
import misclassified_image as mi

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = m.ResNet18().to(device)

classes, trainloader, testloader = d.load()
s.show_random_images(trainloader)

def ep():
  
  

  #use_cuda = torch.cuda.is_available()
  #device = torch.device("cuda" if use_cuda else "cpu")
  print(device)
 # model = m.ResNet18().to(device)
  summary(model, input_size=(3, 32, 32))
  

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=0.0005)
  testLoss = []
  testAcc = []
  EPOCHS = 5
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      print("Device:", device)
      t.train(model, device, trainloader, optimizer, criterion, epoch)
      test_loss , test_acc = t.test(model, device, criterion, testloader)
      testLoss.append(test_loss)
      testAcc.append(test_acc)
      
  import misclassified_image as mi
  mi.show_misclassified_images(model,device, testloader, classes)

  import grad_cam as g 
  g.grad()