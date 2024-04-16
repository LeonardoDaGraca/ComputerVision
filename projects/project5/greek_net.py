'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: Learning on Greek Letters
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

'''
greek data set transform:
Transforms hand-written greek letters to
match MNIST style
'''
class GreekTransform:
    def __init__(self):
        pass
    
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)
    

'''
DataLoader for the Greek data set
'''
def greek_loader(b_size: int, shuffle: bool, training_set_path: str):
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                    GreekTransform(),
                                                                                    torchvision.transforms.Normalize(
                                                                                    (0.1307,), (0.3081,) ) ] ) ),
        batch_size = b_size,
        shuffle = shuffle
    )
    return greek_train

'''
Helper function to calculate accuracy
after training network
'''
def calculate_accuracy(loader, network):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            outputs = network(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def calculate_metrics(loader, network):
    num_classes = 3
    class_correct = [0. for _ in range(num_classes)]
    class_total = [0. for _ in range(num_classes)]
    with torch.no_grad():
        for data, target in loader:
            outputs = network(data)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    #now we calculate precision, recall, and F1 for each class
    for i in range(num_classes):
        precision = class_correct[i] / sum([predicted[j] == i and target[j] == i for j in range(len(target))]) if sum([predicted[j] == i and target[j] == i for j in range(len(target))]) > 0 else 0
        recall = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f'Class {i} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}')


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
    network.fc2 = nn.Linear(num_features, 3)
    
    
    #load greek dataset
    greek_data = greek_loader(5, True, './greek_train/')
    
    #use drawings as a test set
    greek_test = greek_loader(5, False, './greek_drawings')
    
    
    
    '''
    train the model
    '''
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    
    #arrays to track loss and accuracy
    n_epochs = 5
    train_losses, train_counter, train_accuracy = [], [], []
    test_losses, test_accuracy = [], []
    
   
    for epoch in range(1, n_epochs + 1):
        tr_losses, tr_counter, tr_accuracy = train_network(epoch, greek_data, network, optimizer, 0.01, 0.5, 10)
        train_losses.extend(tr_losses)
        train_counter.extend(tr_counter)
        train_accuracy.extend(tr_accuracy)
        test_loss, test_acc = test_network(greek_test, network)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
    
        accuracy = calculate_accuracy(greek_data, network)
        tst_accuracy = calculate_accuracy(greek_test, network)
        print(f'Epoch: {epoch}, Accuracy: {accuracy}%')
        print(f'Epoch: {epoch}, Accuracy: {tst_accuracy}%')
    
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # #plot for training loss
    # ax1.plot(range(1, n_epochs + 1), [train_losses[x - 1] for x in range(1, n_epochs + 1)], color='violet')
    # ax1.scatter(range(1, n_epochs + 1), [train_losses[x - 1] for x in range(1, n_epochs + 1)], color='blue')
    # ax1.legend(['Train Loss'], loc='upper right')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Negative log likelihood loss')

    # # Plot for training accuracy
    # ax2.plot(range(1, n_epochs + 1), train_accuracy, color='violet')
    # ax2.legend(['Train Accuracy'], loc='upper right')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy')

    # plt.tight_layout()
    # plt.show()
    
    fig, axis = plt.subplots(2)
    axis[0].plot(train_counter, train_losses, color='blue')
    axis[0].scatter([x * len(greek_data.dataset) for x in range(1, n_epochs + 1)], test_losses, color='red')
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
        
    print(network)
    


if __name__ == "__main__":
    main(sys.argv)