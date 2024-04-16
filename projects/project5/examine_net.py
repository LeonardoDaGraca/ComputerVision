'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: Examine Network
'''

import sys
import torch
import cv2
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prep_net import MyNetwork, train_MNIST_loader



'''
Main function that reads in the network and 
examines the layers of the network. It then
shows the effects of the filters in a plot.
'''
def main(argv):
    #read in trained network and print model
    network = MyNetwork()
    network_state_dict = torch.load('./results/model_epoch_5')
    network.load_state_dict(network_state_dict)

    #print(network)
    
    #analyze layers
    weights = network.conv_layer.weight.data
    print(f'Shape of the wights tensor: {weights.shape}')
    
    
    
    #print filter weights
    if weights.shape == (10, 1, 5, 5):
        for i in range(10):
            print(f'Filter {i + 1} weights: {weights[i, 0]}')
    else:
        print("Not expected shape for weights tensor")
        
    
    # #3x4 grid to plot the filters
    # fig, axis = plt.subplots(3, 4, figsize=(10, 8))
    # for i, ax in enumerate(axis.flat):
    #     if i < 10:
    #         ax.imshow(weights[i, 0])
    #         ax.set_title(f'Filter {i + 1}')
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     else:
    #         ax.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    #show the effect of the filters
    train_loader = train_MNIST_loader(1, False)
    img, _ = next(iter(train_loader))
    #convert to numpy array and remove batch dimension
    img = img.squeeze().numpy()
    
    #plot
    fig, axis = plt.subplots(5, 4, figsize=(8, 10))
    
    #apply each filter to img
    with torch.no_grad():
        for i in range(10):
            #get filter
            filter_weights = network.conv_layer.weight[i, 0].numpy()
            #apply filter
            filtered_img = cv2.filter2D(img, -1, filter_weights)
            
            #plotting the filter
            axis[i // 2, 2 * (i % 2)].imshow(filter_weights, cmap='gray')
            axis[i // 2, 2 * (i % 2)].set_title(f'Filter {i}')
            axis[i // 2, 2 * (i % 2)].set_xticks([])
            axis[i // 2, 2 * (i % 2)].set_yticks([])

            #plotting the effect of the filter
            axis[i // 2, 2 * (i % 2) + 1].imshow(filtered_img, cmap='gray')
            axis[i // 2, 2 * (i % 2) + 1].set_title(f'Effect of Filter {i}')
            axis[i // 2, 2 * (i % 2) + 1].set_xticks([])
            axis[i // 2, 2 * (i % 2) + 1].set_yticks([])

        #hide unused subplots
        for j in range(20):
            if j >= 10 and j % 2 == 0:
                axis[j // 4, j % 4].axis('off')
            
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main(sys.argv)