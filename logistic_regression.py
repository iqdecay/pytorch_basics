import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from utils import display_losses
from tqdm import tqdm
import matplotlib.pyplot as plt

# Generate toy example
n_features = 2
n_examples = 4000
n_test = 1000
n_epoch = 50


# Function for labeling the toy dataset
def label(x):
    # Return 1 if the point is above the y = 2t - 0.2 line, 0 if not
    if x[1] > 2 * x[0] - 0.2:
        return 1
    else:
        return 0


# Training data
X1_train = torch.randn(n_examples // 2, 2)
X2_train = torch.randn(n_examples // 2, 2) + 1.5
X_train = torch.cat([X1_train, X2_train], dim=0)

Y1_train = torch.zeros(n_examples // 2, 1)
Y2_train = torch.ones(n_examples // 2, 1)
Y_train = torch.cat((Y1_train, Y2_train), dim=0)
plt.scatter(X1_train[:, 0], X1_train[:, 1], color='b')
plt.scatter(X2_train[:, 0], X2_train[:, 1], color='r')
plt.show()
plt.close()

train_data = TensorDataset(Tensor(X_train), Tensor(Y_train))
train_loader = DataLoader(dataset=train_data, batch_size=16)

# Testing data
X1_test = torch.randn(n_test // 2, 2)
X2_test = torch.randn(n_test // 2, 2) + 1.5
X_test = torch.cat((X1_test, X2_test), dim=0)

Y1_test = torch.zeros(n_test // 2, 1)
Y2_test = torch.ones(n_test // 2, 1)
Y_test = torch.cat([Y1_test, Y2_test], dim=0)

test_data = TensorDataset(Tensor(X_test), Tensor(Y_test))
test_loader = DataLoader(dataset=test_data, batch_size=16)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.linear(x)
        return output


model = LogisticRegression(n_features)
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
        optimizer.zero_grad()
        out = model(x)
        loss = loss_function(out, y.squeeze().long())
        loss_sum += loss.item() * x.size(0)
        loss.backward()
        optimizer.step()
    # Average loss over the whole epoch
    train_loss = loss_sum / n_examples
    train_losses[epoch] = train_loss

    # Validation : we don't want to accumulate gradients
    with torch.set_grad_enabled(False):
        loss_sum = 0
        for (x, y) in test_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_function(out, y.squeeze().long())
            loss_sum += loss.item() * x.size(0)
    # Average loss over the whole epoch
    validation_loss = loss_sum / n_test
    validation_losses[epoch] = validation_loss
display_losses(train_losses, validation_losses)
