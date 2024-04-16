'''
Leo DaGraca
CS5330
3/21/2024

MNIST digit recognition: This file is intended to prepare the network 
by training the network on the MNIST data.
'''
import sys
import torch
import torchvision
import torchviz
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#class definitions
'''
A convolution layer with 10 5x5 filters
A max pooling layer with a 2x2 window and a ReLU function applied.
A convolution layer with 20 5x5 filters
A dropout layer with a 0.5 dropout rate (50%)
A max pooling layer with a 2x2 window and a ReLU function applied
A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function on the output
A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
'''
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv_layer = nn.Conv2d(1, 10, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_layer2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
                
    
    #computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_layer(x), 2))
        x = F.relu(F.max_pool2d(self.conv_layer2_drop(self.conv_layer2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    



'''
Function to load in the MNIST train dataset
Parameters: batch size (int), shuffle (bool)
'''
def train_MNIST_loader(b_size: int, shuffle: bool):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
        batch_size=b_size, shuffle=shuffle
    )
    return train_loader

'''
Function to load in the MNIST test dataset
Parameters: batch size (int), shuffle (bool) should be kept to False
'''
def test_MNIST_loader(b_size: int, shuffle: bool):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
        batch_size=b_size, shuffle=shuffle
    )
    return test_loader

'''
Function to plot data
Parameters: size of inputs to be plotted, title of the plot,
            data being passed in, targets
'''
# def plot_data(size: int, title: str, data: any, targets: any):
#     for i in range(size):
#         plt.subplot(2, 3, i + 1)
#         plt.tight_layout()
#         plt.imshow(data[i][0], cmap='gray', interpolation='none')
#         plt.title(f"{title}: {targets[i]}")
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()



'''
Function to train the network
Parameters: epoch, dataset passed in, model, optimizer, learning rate, momentum, log interval
'''
def train_network(epoch: int, t_loader: any, network: any, optimizer: any ,learning_rate: float, momentum: float, log_interval: int):
    network.train()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    t_losses = []
    t_counter = []
    t_accuracy = []
    total_correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(t_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(t_loader.dataset),
                100. * batch_idx / len(t_loader), loss.item()))
            t_losses.append(loss.item())
            t_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(t_loader.dataset))
            )
    accuracy = 100. * total_correct / total
    t_accuracy.append(accuracy)
    
    return t_losses, t_counter, t_accuracy


'''
Function to train the network
Parameters: test dataset, network
'''
def test_network(test_loader: any, network: any):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / total
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, test_accuracy
            

'''
Main function that will train the model based on pre-defined hyper 
parameters. The model will be saved after each epoch. A diagram of the model
will be saved and a plot of the accuracy scores will be displayed.
'''
def main(argv):
    #hyper parameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    
    train_loader = train_MNIST_loader(batch_size_train, True)
    test_loader = test_MNIST_loader(batch_size_test, False)
    
    #view examples
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    
    
    #plot data
    #plot_data(6, "Ground Truth", example_data, example_targets)
    
    '''
    train the model
    '''
    #network and optimizer
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    #arrays to track loss and accuracy
    train_losses, train_counter, train_accuracy = [], [], []
    test_losses, test_accuracy = [], []
    
    
    for epoch in range(1, n_epochs + 1):
        tr_losses, tr_counter, tr_accuracy = train_network(epoch, train_loader, network, optimizer, learning_rate, momentum, log_interval)
        train_losses.extend(tr_losses)
        train_counter.extend(tr_counter)
        train_accuracy.extend(tr_accuracy)
        
        test_loss, test_acc = test_network(test_loader, network)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        
        #save model after each epoch
        torch.save(network.state_dict(), f'./results/model_epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'./results/optimizer_epoch_{epoch}')
    
    
    #draw diagram
    y = network(example_data[1])
    torchviz.make_dot(y, params=dict(list(network.named_parameters()))).render("my_model", format="png")
    
    #plotting
    fig, axis = plt.subplots(2)
    axis[0].plot(train_counter, train_losses, color='blue')
    axis[0].scatter([x * len(train_loader.dataset) for x in range(1, n_epochs + 1)], test_losses, color='red')
    axis[0].legend(['Train Loss', 'Test Loss'], loc='upper right')
    axis[0].set_xlabel('Number of training examples seen')
    axis[0].set_ylabel('Negative log likelihood loss')
    
    axis[1].plot(range(1, n_epochs + 1), train_accuracy, color='blue')
    axis[1].plot(range(1, n_epochs + 1), test_accuracy, color='red')
    axis[1].legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main(sys.argv)
    