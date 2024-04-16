'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: Network Experimentation
'''

import sys
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchviz
from prep_net import train_network, train_MNIST_loader, test_MNIST_loader, test_network

'''
Class definition for experiment network:
Constructor takes in a filter size
input size calculate method to assist with combining layers
'''
class ExperimentNet(nn.Module):
    def __init__(self, filter_size=5):
        super(ExperimentNet, self).__init__()
        self.filter_size = filter_size
        self.conv_layer = nn.Conv2d(1, 10, kernel_size=self.filter_size)
        self.conv_layer2 = nn.Conv2d(10, 20, kernel_size=self.filter_size)
        self.conv_layer2_drop = nn.Dropout2d()

        #calculate the size of the output from the last conv layer to determine
        #the input size for the fully connected layer
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 50)
        self.fc2 = nn.Linear(50, 10)

    def calculate_fc1_input_size(self):
        #zero tensor that simulates the input image
        size_checker = torch.zeros((1, 1, 28, 28))

        #apply the same operations as in the forward pass
        size_checker = F.max_pool2d(self.conv_layer(size_checker), 2)
        size_checker = F.max_pool2d(self.conv_layer2(size_checker), 2)

        #return the resulting flattened size
        return size_checker.view(-1).size(0)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv_layer(x), 2))
        x = F.relu(F.max_pool2d(self.conv_layer2_drop(self.conv_layer2(x)), 2))
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
'''
Helper function to plot the results of the model
'''
def plot_results(results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10)) 
    titles = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']

    #iterate over each subplot and fill it with data
    for i, (ax, title) in enumerate(zip(axs.flat, titles)):
        #initialize lists to hold the x labels and y values
        x_labels = []
        y_values = []
        
        #extract data for each configuration and plot
        for config, metrics in results.items():
            filter_size, batch_size, epoch = config.split('_')
            label = f"F{filter_size} B{batch_size} E{epoch}"
            x_labels.append(label)
    
            #depending on the subplot, use different metrics
            if i == 0:  #Train Loss
                y_values.append(metrics['train_loss'][-1])  #final training loss
            elif i == 1:  #Train Accuracy
                y_values.append(metrics['train_accuracy'][-1])  #final training accuracy
            elif i == 2:  #Test Loss
                y_values.append(metrics['test_loss'])
            elif i == 3:  #Test Accuracy
                y_values.append(metrics['test_accuracy'])
        
        #plot the data
        ax.bar(x_labels, y_values)
        ax.set_title(title)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    #adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


'''
Main function that has the predefined dimensions that
will be evaluated. The 3 dimensions will be evaluated in
a round-robin fashion and the results will then plotted.
'''
def main(argv):
    #ranges for each parameter
    filter_sizes = [3, 5, 7]
    epochs = [5, 10, 15]
    batch_sizes = [32, 64, 128]
    

    results = {}
    
    
    
    #iterate in round robin fashion
    for i in range(max(len(filter_sizes), len(epochs), len(batch_sizes))):
        #use modulus to ensure we cycle through each list
        filters = filter_sizes[i % len(filter_sizes)]
        batch_size = batch_sizes[i % len(batch_sizes)]
        epoch = epochs[i % len(epochs)]

        # Initialize the network and optimizer for each set of parameters
        network = ExperimentNet(filter_size=filters)
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
        
        train_loader = train_MNIST_loader(batch_size, True)
        test_loader = test_MNIST_loader(batch_size, False)
        
        tr_losses, tr_counter, tr_accuracy = train_network(epoch, train_loader, network, optimizer, 0.01, 0.5, 10)
        test_loss, test_acc = test_network(test_loader, network)
        
        dict_key = f'{filters}_{batch_size}_{epoch}'
        results[dict_key] = {
            'train_loss': tr_losses,
            'train_accuracy': tr_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
                
    plot_results(results)
    
    # #view examples
    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # #draw diagram
    # y = network(example_data[1])
    # torchviz.make_dot(y, params=dict(list(network.named_parameters()))).render("experiment_model", format="png")

if __name__ == '__main__':
    main(sys.argv)