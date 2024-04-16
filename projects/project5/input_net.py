'''
Leo DaGraca
CS5330
3/25/2024

MNIST digit recognition: Test the network on new inputs
'''

import sys
import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from prep_net import MyNetwork


'''
Main function that reads in network and then image directory
as a command line argument. The images are then transformed to 
match MNIST style and trained using the network. Results are then
plotted.
'''
def main(argv):
    #network variables & parameters
    network = MyNetwork()
    network_state_dict = torch.load('./results/model_epoch_5')
    network.load_state_dict(network_state_dict)
    network.eval()
    
    #image directory
    img_dir = argv[1]
    images, labels = [], []
    
    for file in os.listdir(img_dir):
        if file.endswith(".png"):
            #read image
            img_path = os.path.join(img_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            #resize image to 28x28
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            
            #change intensity
            if np.mean(img) > 127:
                img = 255 - img
            
            #normalize image
            img = img / 255.0
            img = (img - 0.1307) / 0.3081
            
            
            #convert to a pytorch tensor and add batch dimension
            tensor_img = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
            
            #get prediction
            with torch.no_grad():
                output = network(tensor_img)
            
            predicted_label = output.argmax().item()
            print(f'Processed {file}: Predicted Label: {predicted_label}')
            
            #store images and labels
            images.append(img)
            labels.append(predicted_label)
    
    #plotting the images and their results
    num_images = len(images)
    cols = 5
    rows = num_images // cols + (num_images % cols > 0)
    fig, axis = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    for i, ax in enumerate(axis.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray', interpolation='none')
            ax.set_title(f'Predicted: {labels[i]}')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: program.py [directory]")
    else:
        main(sys.argv)