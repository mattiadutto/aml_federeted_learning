# %% [markdown]
# # Baseline implementation

# %%
# download the Cifar10 non-iid splits, if not present

from os import path
import urllib.request
import zipfile

if not path.exists("cifar10"):
    save_path = "cifar10.zip"
    urllib.request.urlretrieve("http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar10_v1.1.zip", save_path)
    
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall("cifar10")

# %%
import numpy as np

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
config = {
    "E": 1, # number of local epochs
    "K": 5, # number of clients selected each round # [5, 10, 20]
    "NUMBER_OF_CLIENTS": 100, # total number of clients
    "MAX_TIME": 2500,
    "BATCH_SIZE": 50,
    "VALIDATION_BATCH_SIZE": 500,
    "LR": 0.01,
    "DATA_DISTRIBUTION": "non-iid", # "iid" | "non-iid"
    "DIRICHELET_ALPHA": [0.00, 0.05, 0.10],
    "AVERAGE_ACCURACY": np.zeros(8),
    "FED_AVG_M": False,
    "FED_AVG_M_BETA": 0.9,
    "FED_AVG_M_GAMMA": 1,
    "LR_DECAY": 0.99,
    "LOG_FREQUENCY": 25,
    "AUGMENTATION_PROB": 0.0,
    "SAVE_FREQUENCY": 100
}

# %%
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# From: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class Net(nn.Module):

    def __init__(self, *, input_size=32):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        
        # output of the conv layer is (w', h') = (w - 5 + 1, h - 5 + 1)
        # max_pool2d halves the dimensions (w', h') = (w / 2, h / 2)

        # dynamically compute the image size
        size = input_size // 4 - 3
        self.fc1 = nn.Linear(64 * (size * size), 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
#print(net)

# %%
import torch.optim as optim

class Client():
  def __init__(self, i, train_set, validation_set, *, input_size=32):
    self.i = i
    self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["BATCH_SIZE"],
                                         shuffle=True, num_workers=0, pin_memory = True)
    self.validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=config["VALIDATION_BATCH_SIZE"],
                                         shuffle=False, num_workers=0)
    self.net = Net()
    self.net = self.net.to(device)
    # create your optimizer
    self.optimizer = optim.SGD(self.net.parameters(), lr=config["LR"], weight_decay=4e-4)
    self.criterion = nn.CrossEntropyLoss(reduction="sum")
    # wandb.watch(self.net, criterion=self.criterion, log_freq=100, log_graph=False)
    
  def clientUpdate(self, lr, parameters):
    self.net.load_state_dict(parameters)
    self.net.train()

    for g in self.optimizer.param_groups:
      g['lr'] = lr

    for _ in range(config["E"]):
      epoch_loss, n = 0, 0
      for images, labels in self.train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # in your training loop:
        self.optimizer.zero_grad()   # zero the gradient buffers
        outputs = self.net(images)
        loss = self.criterion(outputs, labels)
        epoch_loss += loss
        n += labels.size(0)
        
        loss = loss / labels.size(0)
        loss.backward()
        # wandb.log({f"client-loss-{self.i}": loss.item()})
        self.optimizer.step()    # Does the update
      epoch_loss = epoch_loss / n

    return_dict = {}
    for (k1, v1), (k2, v2) in zip(parameters.items(), self.net.state_dict().items()):
      return_dict[k1] = v1 - v2
    return epoch_loss, return_dict

  def compute_accuracy(self, parameters):
    self.net.load_state_dict(parameters)
    self.net.eval()

    running_corrects = 0
    loss, n = 0, 0
    for data, labels in self.validation_loader:
        data = data.to(device)
        labels = labels.to(device)

        with torch.no_grad():
          outputs = self.net(data)
        loss += self.criterion(outputs, labels).item()

        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data).data.item()
        n += len(preds)
                
    return loss/n, running_corrects / n

# %%
from collections import defaultdict

def parse_csv(filename):
  splits = defaultdict(lambda: [])
  labels_mapping = dict()

  with open(filename) as f:
    for line in f:
      if not line[0].isdigit():
        continue

      user_id, image_id, label = (int(token) for token in line.split(","))
      splits[user_id].append(image_id)
      labels_mapping[image_id] = label

  return splits, labels_mapping

# %%
import time
import json
import numpy
from copy import deepcopy

def listToString(l): 
    return " ".join(str(l))

def printJSON(alpha, acc, net, step = None):
    artifacts_dir = "artifacts"

    artifact_filename = f"ALPHA_{alpha}_E_{config['E']}_K_{config['K']}"
    if step is not None:
      artifact_filename += f"_STEPS_{step}"
      
    # parameters of the trained model
    server_model = net.state_dict()
    # save the model on the local file system
    torch.save(server_model, f"{artifacts_dir}/{artifact_filename}.pth")
    config_copy = deepcopy(config)
    config_copy["DIRICHELET_ALPHA"] = listToString(config_copy["DIRICHELET_ALPHA"])
    config_copy["AVERAGE_ACCURACY"] = numpy.array2string(config_copy["AVERAGE_ACCURACY"])
    data = {
        "config": config_copy,
        "alpha": listToString(alpha),
        "accuracy": acc
    }

    with open(f"{artifacts_dir}/{artifact_filename}.json", "w") as f:
        f.write(json.dumps(data, indent=4))

    # If you want to cat the file, my suggestion is to avoid this is a pretty heavy operation at least on my pc
    #artifact_filename += ".json"
    #!cat artifact_filename


# %%
def selectClients(k):
  return random.sample(clients, k=k)

def aggregateClient(deltaThetas):
  parameters = None
  for i,d in enumerate(deltaThetas):
    #ratio = len(trainsets[i])/len(trainset)
    ratio = len(trainsets[i])/(len(trainsets[i])*config['K'])
    
    if i == 0:
      parameters = {k:ratio*v for k, v in d.items()}
    else:
      for (k, v) in d.items():
        parameters[k] += ratio * v
   
  return parameters

# %%
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from statistics import mean

from tqdm.notebook import tqdm

import os

random.seed(42)

random_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(1),
        transforms.ColorJitter(0.9, 0.9)
    ]
)


trainset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True, 
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomApply([random_transform], config["AUGMENTATION_PROB"]),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]), # from the net, there are the values of cifer10
        ]
    ),
)

testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True, 
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ]
    ),
  )

if not path.exists("artifacts"):
  os.mkdir("artifacts")

# %%
# verify the labels specified in the .csv files are coherent with the actual CIFAR-10 labels
# see https://github.com/google-research/google-research/issues/924

_, labels_mapping = parse_csv(f"cifar10/federated_train_alpha_{0.0:.2f}.csv")
assert(all(label == labels_mapping[idx] for idx, label in enumerate(trainset.targets)))

# %%
for alpha_i, alpha in enumerate(config["DIRICHELET_ALPHA"]):
  net = Net()
  net = net.to(device)

  optimizer = optim.SGD(net.parameters(), lr=config["LR"], momentum=0.9, weight_decay=1e-3)
  scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, min_lr=1e-6, verbose=True)

  if config["DATA_DISTRIBUTION"] == "iid":
    # split the training set
    trainset_len = ( len(trainset) // config["NUMBER_OF_CLIENTS"] ) * config["NUMBER_OF_CLIENTS"]
    trainset = torch.utils.data.Subset(trainset, list(range(trainset_len)))

    lengths = len(trainset) // config["NUMBER_OF_CLIENTS"] * np.ones(config["NUMBER_OF_CLIENTS"], dtype=int)
    trainsets = torch.utils.data.random_split(dataset=trainset, lengths=lengths)
  else:
    dirichelet_splits, _ = parse_csv(f"cifar10/federated_train_alpha_{alpha:.2f}.csv")
    trainsets = [torch.utils.data.Subset(trainset, indices) for indices in dirichelet_splits.values()]


  # split the validation set
  testset_len = ( len(testset) // config["NUMBER_OF_CLIENTS"] ) * config["NUMBER_OF_CLIENTS"]
  testset = torch.utils.data.Subset(testset, list(range(testset_len)))

  lengths = len(testset) // config["NUMBER_OF_CLIENTS"] * np.ones(config["NUMBER_OF_CLIENTS"], dtype=int)
  testsets = torch.utils.data.random_split(dataset=testset, lengths=lengths)


  clientsSizes = torch.zeros(config["NUMBER_OF_CLIENTS"])
  clients = list()



  for c in range(config["NUMBER_OF_CLIENTS"]):
    clients.append(Client(c, trainsets[c], testsets[c]))

  if config["FED_AVG_M"]:
    old_parameters = {}

  # collect the test accuracies over the epochs
  test_accuracies = []

  accuracies = list()

  # best model
  best_model = {}
  best_accuracy = 0.0

  for step in tqdm(range(config["MAX_TIME"])):
    selected_clients = selectClients(config["K"])
    #print(f"Client(s) {[client.i for client in selected_clients]} selected")

    deltaThetas = list()
    losses = list()
    for i, c in enumerate(selected_clients):
      loss, parameters = c.clientUpdate(optimizer.param_groups[0]['lr'], net.state_dict())
      deltaThetas.append(parameters)
      losses.append(loss)
      
    g = aggregateClient(deltaThetas)
    
    parameters = {}
    for (k1, v1), (k2, v2) in zip(net.state_dict().items(), g.items()):
      
      if config["FED_AVG_M"]:
        if k1 in old_parameters:
          parameters[k1] = v1 - config["FED_AVG_M_GAMMA"] * (config["FED_AVG_M_BETA"] * old_parameters[k1] + v2)  
          old_parameters[k1] = config["FED_AVG_M_BETA"] * old_parameters[k1] + v2
        else:
          parameters[k1] = v1 - config["FED_AVG_M_GAMMA"] * v2
          old_parameters[k1] = v2
      else:
        parameters[k1] = v1 - v2 # todo: add server learning rate gamma

    # compute loss and accuracy on the test set of the clients
    # client.compute_accuracy(parameters) returns tuples (loss, accuracy)
    # client_losses_accuracies = [client.compute_accuracy(parameters) for client in clients]
    # client_losses, client_accuracies = zip(*client_losses_accuracies)

    # compute the average client loss
    # and feed it to the scheduler
    # avg_client_loss = mean(client_loss for client_loss in client_losses)
    # scheduler.step(avg_client_loss)

    # compute the average accuracy
    if step % config["LOG_FREQUENCY"] == 0:
      client_losses_accuracies = [client.compute_accuracy(parameters) for client in clients]
      client_losses, client_accuracies = zip(*client_losses_accuracies)
      
      avg_client_accuracy = mean(client_acc for client_acc in client_accuracies)
      accuracies.append(avg_client_accuracy * 100)
      
      if avg_client_accuracy >= best_accuracy:
        best_accuracy = avg_client_accuracy
        best_model = net.state_dict()
          
      print(f"Average accuracy after {step} rounds is {avg_client_accuracy*100}")    

    net.load_state_dict(parameters)

    if step % config["SAVE_FREQUENCY"] == 0:
      printJSON(alpha, accuracies, net, step)
  
  avg_accuracy = mean(float(client.compute_accuracy(best_model)[1]) for client in clients)
  #model_parameters = net.state_dict()
  #avg_accuracy = mean(float(client.compute_accuracy(model_parameters)[1]) for client in clients)
 
  #alpha = config["DIRICHELET_ALPHA"][i]
  config["AVERAGE_ACCURACY"][alpha_i] = avg_accuracy
  print(f"Average accuracy with alpha = {alpha} after {step+1} rounds is {avg_accuracy*100}")
  printJSON(alpha, accuracies, net)


# %%
import shutil
shutil.make_archive('artifacts', 'zip', 'artifacts')


