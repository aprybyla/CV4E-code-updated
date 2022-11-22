import os
import argparse
import yaml
import glob
from tqdm import trange
import ipdb 
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn

# let's import our own classes and functions!
from util import init_seed
from dataset import AudioDataset
from model import BeeNet #THIS NEEDS TO BE THE SAME AS BeeNet (custom)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score 

def create_dataloader(cfg, split='test'): #CHANGED THE PATH TO TEST DATA
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = AudioDataset(cfg, split)        # create an object instance of our AudioDataset class
    
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'], #whatever I said was my batch size in the config file, this will be represented here bc it is pulled from that file 
            shuffle=True, #shuffles files read in, which changes them every epoch. Usually turned off for train and val sets. Must do manually. 
            num_workers=cfg['num_workers']
        )
    return dataLoader


# This tells us how to start a model that we have previoussly stopped or paused, and we need to start from the same epoch 
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = BeeNet(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('/datadrive/audio_classifier/model_states-weighted2/*.pt') #this looks for the saved model files
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('/datadrive/audio_classifier/model_states-weighted2/','').replace('.pt','')) for m in model_states] #you can define what epoch you want to start on 
        eval_epoch = '63'

        # load state dict and apply weights to model
        print(f'Evaluating from epoch {eval_epoch}')
        state = torch.load(open(f'/datadrive/audio_classifier/model_states-weighted2/{eval_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

        # MOST OF THIS WILL BE COMMENTED OUT FOR US BECAUSE WE WONT BE STARTING WITH SOMETHING NEW

    else:
        # no save state found; start anew
        print('No model found')
        #start_epoch = 0

    return model_instance, eval_epoch


# THIS IS HOW THE MODEL TRAINS. The validation function is almost the same, some key differences: no backward pass here. We do not run the optimizer here: optimize
# on the training data, but not validate. 
def predict(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    # # iterate over dataLoader
    # progressBar = trange(len(dataLoader))
    
    confidence_prediction_list = []

    with torch.no_grad(): 
        true_labels = []
        predicted_labels = []
        confidences_0 = [] 
        confidences_1 = []
        confidences_2 = []
        confidences_3 = []
        confidences_4 = []
        confidences_5 = []
        confidences_6 = []
        confidences_7 = []
        confidences_8 =[]           
        
        # to - do: add individual  
        # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):
            true_labels.extend(labels)
            # put data and labels on device
            data, labels = data.to(device), labels.to(device)
            
            # forward pass
            prediction = model(data)

            pred_label = torch.argmax(prediction, dim=1).cpu().numpy()
            predicted_labels.extend(pred_label)

            confidence = torch.nn.Softmax(dim=1)(prediction).cpu().numpy() #this is a long confidence probability vector 
            conf_0 = confidence[:,0]
            conf_1 = confidence[:,1]
            conf_2 = confidence[:,2]
            conf_3 = confidence[:,3]
            conf_4 = confidence[:,4]
            conf_5 = confidence[:,5]
            conf_6 = confidence[:,6]
            conf_7 = confidence[:,7]
            conf_8 = confidence[:,8]
            confidence_prediction_list.append(confidence)

            confidences_0.extend(conf_0)
            confidences_1.extend(conf_1)
            confidences_2.extend(conf_2)
            confidences_3.extend(conf_3)
            confidences_4.extend(conf_4)
            confidences_5.extend(conf_5)
            confidences_6.extend(conf_6)
            confidences_7.extend(conf_7)
            confidences_8.extend(conf_8) 

    return predicted_labels, true_labels, confidences_0, confidences_1, confidences_2, confidences_3, confidences_4, confidences_5, confidences_6, confidences_7, confidences_8, confidence_prediction_list


def save_confusion_matrix(true_labels, predicted_labels, cfg):
    # make figures folder if not there

    matrix_path = cfg['data_root']+'/figs'
    #### make the path if it doesn't exist
    if not os.path.exists(matrix_path):  
        os.makedirs(matrix_path, exist_ok=True)

    confmatrix = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confmatrix)
    #confmatrix.save(cfg['data_root'] + '/experiments/'+(args.exp_name)+'/figs/confusion_matrix_epoch'+'_'+ str(split) +'.png', facecolor="white")
    disp.plot()
    plt.savefig(cfg['data_root'] +'/figs/confusion_matrix-model_states-weighted2.png', facecolor="white")
       ## took out epoch)
    return confmatrix

def save_prevision_recall_curve(cfg, true_labels, predicted_labels, confidence_prediction_list): 
    number_of_classes = cfg['num_classes']
    for entry in range (0, number_of_classes):
        # true_labels == entry
        binarized_true_labels = []
        for l in true_labels:
            if l == entry:
                binarized_true_labels.append(1)
            else:
                binarized_true_labels.append(0)
        y_true = np.array(binarized_true_labels)
        y_scores = confidence_prediction_list[:,entry]
        #ipdb.set_trace()
        print(average_precision_score(y_true, y_scores)) 
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores) #recall is x axis, precision is y axis
        plt.plot(recall, precision)
        plt.savefig(cfg['data_root'] +'/figs/PR-curve-class-'+ str(entry) + '-overlay' + '.png', facecolor="white")
        plt.clf()
        #ipdb.set_trace()
    
# When you call train.py, Main is what starts running. It has all of the functions we defined above, and it puts them all in here. When you call the file, it looks 
# through Main, and then when it hits a function, it goes to the to the top to understand the function, and then goes back to Main. 
def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.') # this is how it knows to look at different things 
    parser.add_argument('--config', help='Path to config file', default='configs/exp_resnet18.yaml') # change path to config using https://github.com/CV4EcologySchool/audio_classifier_example and scrolling down. Change on command line when you run. 
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    dl_val = create_dataloader(cfg, split='test') #dl_val means dataloader validation 

    # initialize model
    model, current_epoch = load_model(cfg)

    predicted_labels, true_labels, confidences_0, confidences_1, confidences_2, confidences_3, confidences_4, confidences_5, confidences_6, confidences_7, confidences_8, confidence_prediction_list = predict(cfg, dl_val, model)  
    confidence_prediction_list = np.concatenate(np.array(confidence_prediction_list, dtype='object'))
    bac = balanced_accuracy_score(true_labels, predicted_labels)
    print(bac)

    save_prevision_recall_curve(cfg, true_labels, predicted_labels, confidence_prediction_list)

    confmatrix = save_confusion_matrix(true_labels, predicted_labels, cfg)
    print("confusion matrix saved")

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()