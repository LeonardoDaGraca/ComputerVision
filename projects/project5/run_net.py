'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: File to read network and run it 
on the test set
'''

import sys
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prep_net import MyNetwork, test_MNIST_loader




'''
Main function that will read in network
and run the model on the first 10 test set examples
'''
def main(argv):
    learning_rate = 0.01
    momentum = 0.5
  
    #read network
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    
    network_state_dict = torch.load('./results/model_epoch_5')
    optimizer_state_dict = torch.load('./results/optimizer_epoch_5')
    
    network.load_state_dict(network_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    
    
    #set network to evaluation mode
    network.eval()
    
    #load first 10 examples from test set
    test_loader = test_MNIST_loader(10, False)
    test_examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(test_examples)
    #print(example_data.shape)
    
    #run model on first 10 examples
    with torch.no_grad():
        output = network(example_data)
    
    #printing values for each example image
    for i in range(10):
        print(f'Example {i + 1}:')
        print(f'Output values: {output[i].data.numpy().round(2)}')
        print(f'Predicted label: {output[i].argmax().item()}')
        print(f'Correct label: {example_targets[i].item()}')
        print()
    
    #plot first 9 digits
    fig, axis = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axis.flat):
        ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f'Prediction: {output[i].argmax().item()}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main(sys.argv)