import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

print("Using PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("Using GPU, device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("No GPU found, using CPU instead.")
    device = torch.device("cpu")

# set seed for pseudo-random number generation
torch.manual_seed(0)

# transform to convert images to pytorch tensors and normalize them
base_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# download the training and test datasets
train_dataset = datasets.MNIST(
    "./data", train=True, download=True, transform=base_transform
)
test_dataset = datasets.MNIST(
    "./data", train=False, download=False, transform=base_transform
)

print(f"There are {len(train_dataset)} images in the training dataset.")
print(f"There are {len(test_dataset)} images in the test dataset.")

print(f"The shape of the images is {train_dataset[0][0].shape}.")
print(f"There are {len(train_dataset.targets.unique())} classes.")


def plot_img(img, ax):
    """
    Plot a single image.

    Args:
        img (torch.Tensor): Image to plot.
        ax (matplotlib.axes.Axes): Axes to plot on.

    """
    img = img.numpy()  # convert to np array
    img = img.transpose(1, 2, 0)  # permute dimensions

    ax.imshow(img, cmap="gray", interpolation=None)  # add to plot
    ax.axis("off")


# Function to plot the image
def plot_img(img, ax):
    # Convert tensor to numpy array and plot
    ax.imshow(img.squeeze().numpy(), cmap='gray')
    ax.axis('off')


# axes for plotting
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(axes.ravel()):
    # select a random image
    img = train_dataset[random.randint(0, len(train_dataset) - 1)][0]

    # plot the image
    plot_img(img, ax)

plt.show()


def class_balanced_subset(dataset, subset_size):
    """
    Create a class-balanced subset of a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset.
        subset_size (int): The desired size of the subset.

    Returns:
        torch.utils.data.Subset: The class-balanced subset.

    """

    targets = dataset.targets.unique()
    num_classes = len(targets)

    assert not subset_size % num_classes
    samples_per_class = subset_size // num_classes

    subset_indices = []

    for target in targets:
        target_indices = torch.nonzero(dataset.targets == target)
        subset_indices.append(target_indices[:samples_per_class].flatten())

    subset = Subset(dataset, torch.cat(subset_indices))

    return subset


NUM_TRAIN = 250
NUM_TEST = 1000

train_subset = class_balanced_subset(train_dataset, NUM_TRAIN)
test_subset = class_balanced_subset(test_dataset, NUM_TEST)

print(f"There are {len(train_subset)} images in the training dataset.")
print(f"There are {len(test_subset)} images in the test dataset.")

print(f"The shape of the images is {train_subset[0][0].shape}.")
# print(f"There are {len(train_subset.targets.unique())} classes.")

batch_size_train = 16
batch_size_test = len(test_subset) // 10  # for easy accurate accuracy calculation

train_loader = DataLoader(train_subset, batch_size_train, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size_test, shuffle=False, num_workers=0)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.
    """

    def __init__(self, input_shape=[1, 28, 28], sz_hids=[1024], num_classes=10):
        """
        Initialize the MLP model.

        Args:
            input_shape (tuple): Input shape, default is (1, 28, 28)
            sz_hids (list): List of hidden layer sizes, default is [1024].
            num_classes (int): Number of output classes, default is 10.

        There should be len(sz_hids) hidden layers, hence (len(sz_hids) + 1) linear layers.

        """
        super().__init__()

        # SOLUTION START ###

        input_size = input_shape[0] * input_shape[1] * input_shape[2]   # Calculate the number of input features

        self.fc1 = nn.Linear(input_size, sz_hids[0])
        # self.fc2 = nn.Linear(sz_hids[0], sz_hids[0]//2)
        self.ReLU = nn.ReLU()
        self.fc3 = nn.Linear(sz_hids[0], num_classes)

        # SOLUTION END ###

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        # SOLUTION STARTS HERE ###

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.ReLU(x)
        # x = self.fc2(x)
        x = self.fc3(x)

        # SOLUTION ENDS HERE ###

        return x


def get_acc(outputs, labels):
    """
    Calculate the accuracy of the model predictions.

    Args:
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        torch.Tensor: Accuracy of the model predictions.

    """
    return (outputs.argmax(dim=1) == labels).sum().float() / outputs.shape[0]


def train_epoch(model, dataloader, loss_fn, opt):
    """
    Perform a single training epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.DataLoader): The dataloader containing the training data.
        loss_fn (Callable): The loss function to calculate the loss.
        opt (torch.optim.Optimizer): The optimizer to update the model parameters.

    Returns:
        float: Mean loss value.
        float: Mean accuracy value.

    """

    model.train()

    losses = []
    accs = []

    for data, target in dataloader:
        opt.zero_grad()  # zero gradients in the optimizer
        data, target = data.to(device), target.to(device)

        output = model(data)  # forward pass through the model
        loss = loss_fn(output, target)  # calculate the loss

        loss.backward()  # backpropagate the loss
        opt.step()  # update the model parameters

        losses.append(loss.item())
        accs.append(get_acc(output, target).item())

    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)

    return mean_loss, mean_acc


# decorator to disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    """
    Evaluate the model on the given dataloader.
    Args:

        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the evaluation data.
        loss_fn (Callable): The loss function to calculate the loss.

    Returns:
        float: Mean loss value.
        float: Mean accuracy value.

    """

    model.eval()

    losses = []
    accs = []

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        output = model(data)  # forward pass through the model
        loss = loss_fn(output, target)  # calculate the loss

        losses.append(loss.item())
        accs.append(get_acc(output, target).item())

    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)

    return mean_loss, mean_acc


def training_run(model, train_loader, test_loader, num_epochs=40, patience=10):
    """
    Perform a full training run.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The dataloader containing the training data.
        test_loader (torch.utils.data.DataLoader): The dataloader containing the test data.
        num_epochs (int): The number of training epochs.
        patience (int): The number of epochs to wait for improvement in test accuracy before early stopping.
    """

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    best_test_acc = -1
    patience_counter = 0

    for i in range(num_epochs):
        # train the model for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, opt)

        # evaluate the model on the test data
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)

        # check if the current test accuracy is better than the best test accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc

            # add a message indicating a new best test accuracy
            msg = "- new best!"

            # reset the patience counter
            patience_counter = 0
        else:
            # empty message if there is no improvement in test accuracy
            msg = ""

            # increment the patience counter
            patience_counter += 1

        # print the training and test metrics for the current epoch
        print(
            f"Epoch {i+1:02d}: train loss {train_loss:.4f} train acc {train_acc:.4f} test loss {test_loss:.4f} test acc {test_acc:.4f} {msg}"
        )

        # if patience_counter == patience:
        #     print(f"No progress after {patience} epochs - stopping early.\n")
        #     break


def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# This is the baseline model of non-translated images below, comment it out when needed.

# model = MLP().to(device)
# print(f"Model contains {count_parameters(model)} trainable parameters\n")
# training_run(model, train_loader, test_loader)


'''
 After Training the model with just 1 Hidden layer, the test accuracy was 79-80%,
 however using an additional hidden layer as shown in the code actually performed
 worse and the process ended before 20 epochs multiple times as there was no 
 improvement in performance on the test sub-dataset. Also initially I ran the MLP
 without the ReLU function this constrained me to the 79-80% but after adding it
 in after the first G-Invariant Linear layer I was able to successfully improve 
 the accuracy up to the desired 81-82%
    
 No ReLU:
 Epoch 40: train loss 0.0006 train acc 1.0000 test loss 0.9502 test acc 0.7988 
 
 With ReLU:
 Epoch 40: train loss 0.0004 train acc 1.0000 test loss 0.7706 test acc 0.8181 
 
 The next thing to do in our code now that we have a baseline value for the MLP
 models accuracy is to augment our data using random degrees of translations to
 our input training samples
'''

# the new transform is the composition of our random translation and the base transform
translate_transform = transforms.Compose(
    [transforms.RandomCrop((28, 28), 2), base_transform]
)

# new train_subset with the augmented transform
train_dataset_translate = datasets.MNIST(
    "./data", train=True, download=True, transform=translate_transform
)
train_subset_translate = class_balanced_subset(train_dataset_translate, NUM_TRAIN)

# new train_loader
train_loader_translate = DataLoader(
    train_subset_translate, batch_size_train, shuffle=True, num_workers=0
)

# Can uncomment to test translated sample

# model = MLP().to(device)
# print(f"Model contains {count_parameters(model)} trainable parameters\n")
# training_run(model, train_loader_translate, test_loader)


'''
 After multiple attempts in running the code with the augmented/translated data the model
 consistently performs worse on the original test sample with a typical accuracy score of
 around 70-76%, this was without the ReLU and is important to note how that with the ReLU
 we can experimentally verify that having a pooling layer to coarse the data is important
 as the import features get highlighted in the model.
 
 No ReLU:
 Epoch 40: train loss 0.7034 train acc 0.7742 test loss 0.9580 test acc 0.7163 
 
 With ReLU:
 Epoch 39: train loss 0.0857 train acc 0.9680 test loss 0.5010 test acc 0.8625 - new best!
 Epoch 40: train loss 0.0543 train acc 0.9805 test loss 0.5433 test acc 0.8458 
'''


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model.
    """

    def __init__(
        self, input_shape=(1, 28, 28), feature_dims=[64, 64, 32], num_classes=10
    ):
        """
        Initialize the CNN model.

        Args:
            input_shape (list): The shape of the input tensor, default is [1, 28, 28].
            feature_dims (list): List of feature dimensions, default is [64, 64, 32].
            num_classes (int): Number of output classes, default is 10.

        There should be len(feature_dims) convolutional layers and a single linear layer.

        """

        super().__init__()

        # SOLUTION START

        self.conv1 = nn.Conv2d(input_shape[0], feature_dims[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_dims[0], feature_dims[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_dims[1], feature_dims[2], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.ReLU = nn.ReLU()

        self.fc_input_size = feature_dims[2] * 7 * 7  # Example size after pooling and convolutions in this project
        self.fc1 = nn.Linear(self.fc_input_size, num_classes)  # Output size is 10 for a 10-class classification problem

        # SOLUTION END

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        # SOLUTION START

        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = self.ReLU(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # SOLUTION END

        return x


model = CNN().to(device)
img = next(iter(test_loader))[0].to(device)
output = model(img)
assert output.shape == (batch_size_test, 10)

model = CNN().to(device)
print(f"Model contains {count_parameters(model)} trainable parameters\n")
training_run(model, train_loader, test_loader)

model = CNN().to(device)
print(f"Model contains {count_parameters(model)} trainable parameters\n")
training_run(model, train_loader_translate, test_loader)

'''
 After using the CNN architecture we notice two things of importance. Firstly, the test accuracy increases under the 
 training set being translated. This is shown below:
 
 Original-dataset:
 Epoch 40: train loss 0.0002 train acc 1.0000 test loss 0.7639 test acc 0.8711 
 
 Translated dataset:
 Epoch 40: train loss 0.0269 train acc 0.9961 test loss 0.3759 test acc 0.9048 
 
 Secondly, leading on from the first point we notice that the training set which is invariant under translations does 
 affect the learning function. This can be due to multiple reasons such as: 1) Increased generalisation and better 
 feature extraction - this means that the positions of the shapes have less weighting in the models decision but rather
 the more general features of a number is now more prominent as to why the model picks the number it does. 2) Reduced 
 overfitting, by translating the input images specific positions in the feature map aren't as overfitted. 3) Robustness
 - the model becomes more robust to translations in the training data and is better suited to analyse this parameter.
'''