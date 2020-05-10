import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import display_losses

#  Load and normalize dataset
transform_function = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
mnist_train = torchvision.datasets.MNIST("data", train=True,
                                         transform=transform_function,
                                         download=True)
mnist_test = torchvision.datasets.MNIST("data", train=False,
                                        transform=transform_function,
                                        download=True)

n_epoch = 10
n_examples = len(mnist_train)
n_test = len(mnist_test)
batch_size = 64

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=batch_size)


class MNISTClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MNISTClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, 100)
        # self.linear2 = nn.Linear(1000, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        x1 = self.linear1(x)
        # x2 = self.linear2(x1)
        x3 = self.linear3(x1)
        return x3


# The [1, 28, 28] images will be flattened to [784] tensors
model = MNISTClassifier(28 * 28)
loss_function = nn.CrossEntropyLoss()
# Use SGD because it is an easier algorithm to understand, which fits with a
# simple example such as this
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_losses = {}
validation_losses = {}
for epoch in tqdm(range(n_epoch)):
    loss_sum = 0
    # Training
    for i, (x, y) in enumerate(train_loader):
        # Flatten the input from [1, 28,28] images to [784] tensor
        x = x.view(x.shape[0], -1)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_function(out, y)
        loss.backward()
        loss_sum += loss.item() * x.size(0)
        optimizer.step()
    # Average loss over the whole epoch
    train_loss = loss_sum / n_examples
    train_losses[epoch] = train_loss

    # Validation : we don't want to accumulate gradients
    with torch.set_grad_enabled(False):
        loss_sum = 0
        for i, (x, y) in enumerate(test_loader):
            optimizer.zero_grad()
            x = x.view(x.shape[0], -1)
            out = model(x)
            loss = loss_function(out, y)
            loss_sum += loss.item() * x.size(0)
    # Average loss over the whole epoch
    validation_loss = loss_sum / n_test
    validation_losses[epoch] = validation_loss

display_losses(train_losses, validation_losses)
