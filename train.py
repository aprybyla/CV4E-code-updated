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

# let's import our own classes and functions!
from util import init_seed
from dataset import AudioDataset
from model import BeeNet #THIS NEEDS TO BE THE SAME AS BeeNet (custom)
from sklearn.utils import class_weight
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score 


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = AudioDataset(cfg, split)        # create an object instance of our AudioDataset class

    classes_for_weighting = [] # This is a list containing the labels for every sample in our dataset 

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'], #whatever I said was my batch size in the config file, this will be represented here bc it is pulled from that file 
            shuffle=True, #shuffles files read in, which changes them every epoch. Usually turned off for train and val sets. Must do manually. 
            num_workers=cfg['num_workers']
        )
    for data, labels in dataLoader:
        classes_for_weighting.extend(list(labels.numpy()))  

    class_weights=class_weight.compute_class_weight('balanced',classes = np.unique(classes_for_weighting),y = np.array(classes_for_weighting))
    class_weights = class_weights/np.sum(class_weights)
    class_weights=torch.tensor(class_weights,dtype=torch.float).cuda()

    return dataLoader, class_weights


# This tells us how to start a model that we have previoussly stopped or paused, and we need to start from the same epoch 
def load_model(cfg):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = BeeNet(cfg['num_classes'])         # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob('model_states-prcurve/*.pt') #this looks for the saved model files
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states-prcurve/','').replace('.pt','')) for m in model_states] #you can define what epoch you want to start on 
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states-prcurve/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

        # MOST OF THIS WILL BE COMMENTED OUT FOR US BECAUSE WE WONT BE STARTING WITH SOMETHING NEW

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


# This is how we save the model states that we previously talked about. When it is running, this says it is making a directory for where to save them. 
def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states-species', exist_ok=True) #don't overwrite 

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states-prcurve/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states-prcurve/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)

            
# cfg means that this file is going to the config file, and this is how to optimize the training process
def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = SGD(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


# THIS IS WHAT GETS THE MODEL TO TRAIN. Says what device to put it on (GPU), &c... 
def train(cfg, dataLoader, model, optimizer, classes_for_weight_train):
    '''
        Our actual training function.
    '''

    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    criterion = nn.CrossEntropyLoss(classes_for_weight_train)

    # running averages (define the loss, oa is overall accuracy)
    loss_total, oa_total = 0.0, 0.0                         # for now, we just log the loss and overall accuracy (OA)

    all_predicted_labels = []
    all_ground_truth_labels = []

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    for idx, (data, labels) in enumerate(dataLoader):       # see the last line of file "dataset.py" where we return the image tensor (data) and label

        # Loads all of these labels and things onto the same device, the GPU 
        # put data and labels on device
        data, labels = data.to(device), labels.to(device)

        # forward pass
        prediction = model(data)
        torch.nn.functional.softmax(prediction, dim=1, _stacklevel=3, dtype=None)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        loss_total += loss.item()                       # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor

        pred_label = torch.argmax(prediction, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
        all_ground_truth_labels.extend(labels.cpu()) # this moves all ground truth labels to the list above
        oa = torch.mean((pred_label == labels).float()) # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)           # shorthand notation for: loss_total = loss_total / len(dataLoader)
    oa_total /= len(dataLoader)
    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)

    return loss_total, oa_total, bac


# THIS IS HOW THE MODEL TRAINS. The validation function is almost the same, some key differences: no backward pass here. We do not run the optimizer here: optimize
# on the training data, but not validate. 
def validate(cfg, dataLoader, model, classes_for_weight_val):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    criterion = nn.CrossEntropyLoss(classes_for_weight_val)   # we still need a criterion to calculate the validation loss

    all_predicted_labels = []
    all_ground_truth_labels = []

    # running averages
    loss_total, oa_total = 0.0, 0.0     # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    with torch.no_grad():               # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
        for idx, (data, labels) in enumerate(dataLoader):

            # put data and labels on device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            all_predicted_labels.extend(pred_label.cpu()) # this moves all predicted labels to a list above
            all_ground_truth_labels.extend(labels.cpu()) # this moves all ground truth labels to the list above
            oa = torch.mean((pred_label == labels).float())

            oa_total += oa.item()

            progressBar.set_description(
                '[Val ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    
    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(dataLoader)
    bac = balanced_accuracy_score(all_ground_truth_labels, all_predicted_labels)
    oa_total /= len(dataLoader)

    # unique_gt_labels = list(set(all_ground_truth_labels.cpu()))
    # all_ground_truth_labels = np.array(all_ground_truth_labels.cpu())
    # all_predicted_labels = np.array(all_predicted_labels.cpu())

    # for element in unique_gt_labels: 
    #     denominator = list(all_ground_truth_labels).count(element)
    #     gtl_element = np.where(all_ground_truth_labels == element)[0]
    #     apl_element = np.where(all_predicted_labels == element)[0]
    #     intersected_matrices = np.intersect1d(gtl_element, apl_element)
    #     numerator = intersected_matrices.shape[0]
    #     print(element)
    #     print(numerator/denominator)

    return loss_total, oa_total, bac


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

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # initialize data loaders for training and validation set
    dl_train, classes_for_weight_train = create_dataloader(cfg, split='train')
    dl_val, classes_for_weighting_val = create_dataloader(cfg, split='val')

    # initialize model
    model, current_epoch = load_model(cfg)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    #tensorboard initialize
    writer = SummaryWriter()

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    # Tells you what epochs you're on, spits out eval metrics, like loss and the overall accuracy of the train and validation data
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        #notice the difference here: there is no optimization in the val set, and no backwards passing
        loss_train, oa_train, bac_train = train(cfg, dl_train, model, optim, classes_for_weight_train)
        loss_val, oa_val, bac_val = validate(cfg, dl_val, model, classes_for_weighting_val)

        writer.add_scalar('Train loss', loss_train, current_epoch)
        writer.add_scalar('Val loss', loss_val, current_epoch)
        writer.add_scalar('Balanced train oa', bac_train, current_epoch)
        writer.add_scalar('Balanced val oa', bac_val, current_epoch)
        writer.flush()

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'bac_train': bac_train,
            'bac_val': bac_val
        }
        save_model(cfg, current_epoch, model, stats)
        writer.close()

    # That's all, folks!
        


if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()