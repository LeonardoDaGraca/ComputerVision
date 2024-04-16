'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: Extension more Greek letters
'''

import sys
import torch
import cv2
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prep_net import MyNetwork, train_network, test_network
from greek_net import calculate_accuracy, greek_loader


'''
Main function that reads in trained network,
freezes network weights, loads greek dataset,
trains greek data set, and plots the results
'''
def main(argv):
    #read in trained network
    network = MyNetwork()
    network_state_dict = torch.load('./results/model_epoch_5')
    network.load_state_dict(network_state_dict)
    
    #freeze network weights
    for param in network.parameters():
        param.requires_grad = False
    
    #replace last layer with a new linear layer with three nodes
    num_features = network.fc2.in_features #number of input features - greek letters
    network.fc2 = nn.Linear(num_features, 5)
    
    
    #load greek dataset
    greek_data = greek_loader(20, True, './greek_test/')
    
    
    
    '''
    train the model
    '''
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    
    #arrays to track loss and accuracy
    n_epochs = 5
    train_losses, train_counter, train_accuracy = [], [], []
    # test_losses, test_accuracy = [], []
    
   
    for epoch in range(1, n_epochs + 1):
        tr_losses, tr_counter, tr_accuracy = train_network(epoch, greek_data, network, optimizer, 0.01, 0.5, 10)
        train_losses.extend(tr_losses)
        train_counter.extend(tr_counter)
        train_accuracy.extend(tr_accuracy)
        # test_loss, test_acc = test_network(greek_test, network)
        # test_losses.append(test_loss)
        # test_accuracy.append(test_acc)
    
        accuracy = calculate_accuracy(greek_data, network)
        # tst_accuracy = calculate_accuracy(greek_test, network)
        print(f'Epoch: {epoch}, Accuracy: {accuracy}%')
        # print(f'Epoch: {epoch}, Accuracy: {tst_accuracy}%')
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    #plot for training loss
    ax1.plot(range(1, n_epochs + 1), [train_losses[x - 1] for x in range(1, n_epochs + 1)], color='violet')
    ax1.scatter(range(1, n_epochs + 1), [train_losses[x - 1] for x in range(1, n_epochs + 1)], color='blue')
    ax1.legend(['Train Loss'], loc='upper right')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative log likelihood loss')

    # Plot for training accuracy
    ax2.plot(range(1, n_epochs + 1), train_accuracy, color='violet')
    ax2.legend(['Train Accuracy'], loc='upper right')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
