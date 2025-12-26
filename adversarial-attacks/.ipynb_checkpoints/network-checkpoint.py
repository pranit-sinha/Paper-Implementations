import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SmallAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallAlexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 384, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(384, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.flat_size = 192 * 8 * 8
        
        self.fc1 = nn.Linear(self.flat_size, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = F.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = F.local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2)
        
        x = x.view(-1, self.flat_size)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

model = SmallAlexNet()
model.load_state_dict(torch.load('model_weight.pt', weights_only=True))
model.eval()

meta = unpickle('cifar-10-batches-py/batches.meta')
labels = meta[b'labels']
labels = [label.decode("utf-8") for label in labels]
classes = tuple(labels)
test_data = unpickle('cifar-10-batches-py/test_batch')
test_images = test_data[b'data'].reshape(10000, 3, 32, 32).astype(np.float32) / 255.0
test_labels = test_data[b'labels']

# cf. https://stackoverflow.com/a/68123869
mean = [0.49139968, 0.48215827 ,0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]
test_images = (test_images - np.array(mean).reshape(1, 3, 1, 1)) / np.array(std).reshape(1, 3, 1, 1)
                                                                                                                                                                                     
testset = torch.utils.data.TensorDataset(torch.tensor(test_images), torch.tensor(labels))
testloader = torch.utils.data.DataLoader(testset, batch_size = 16, shuffle = False, num_workers = 2)


