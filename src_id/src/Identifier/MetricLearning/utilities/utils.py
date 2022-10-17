# Core libraries
import os
import sys
sys.path.append(os.path.abspath("../../"))
import json
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# PyTorch stuff
import torch
from torch import optim
from torch.utils import data

# Import our own classes
from utilities.loss import *
from utilities.mining_utils import *
from models.TripletResnet import TripletResnet50
from Datasets.RGBDCows2020 import RGBDCows2020

"""
File contains a collection of utility functions used for training and evaluation
"""

class Utilities:
    # Class constructor
    def __init__(self, args):
        # Store the arguments
        self.args = args

        # Where to store training logs
        self.log_path = os.path.join(args.fold_out_path, "logs.npz")

        # Prepare arrays to store training information
        self.loss_steps = []
        self.losses_mean = []
        self.losses_softmax = []
        self.losses_triplet = []
        self.accuracy_steps = []
        self.accuracies = []

    # Preparations for training for a particular fold
    def setupForTraining(self, args):
        # Retrieve the correct dataset
        dataset = Utilities.selectDataset(args, "train")

        # Wrap up the data in a PyTorch dataset loader
        data_loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=6, shuffle=True)

        # Setup the selected model
        if args.model == "TripletResnet":
            model = TripletResnet50(pretrained=True, num_classes=dataset.getNumClasses(), img_type=args.img_type)
        else:
            print(f"Model choice: \"{args.model}\" not recognised, exiting.")
            sys.exit(1)

        # Put the model on the GPU and in training mode
        model.cuda()
        model.train()

        # Setup the triplet selection method
        if args.triplet_selection == "HardestNegative":
            triplet_selector = HardestNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "RandomNegative":
            triplet_selector = RandomNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "SemihardNegative":
            triplet_selector = SemihardNegativeTripletSelector(margin=args.triplet_margin)
        elif args.triplet_selection == "AllTriplets":
            triplet_selector = AllTripletSelector()
        else:
            print(f"Triplet selection choice not recognised, exiting.")
            sys.exit(1)

        # Setup the selected loss function
        if args.loss_function == "TripletLoss":
            loss_fn = TripletLoss(margin=args.triplet_margin)
        elif args.loss_function == "TripletSoftmaxLoss":
            loss_fn = TripletSoftmaxLoss(margin=args.triplet_margin)
        elif args.loss_function == "OnlineTripletLoss": 
            loss_fn = OnlineTripletLoss(triplet_selector, margin=args.triplet_margin)
        elif args.loss_function == "OnlineTripletSoftmaxLoss":
            loss_fn = OnlineTripletSoftmaxLoss(triplet_selector, margin=args.triplet_margin)
        elif args.loss_function == "OnlineReciprocalTripletLoss":
            loss_fn = OnlineReciprocalTripletLoss(triplet_selector)
        elif args.loss_function == "OnlineReciprocalSoftmaxLoss":
            loss_fn = OnlineReciprocalSoftmaxLoss(triplet_selector)
        else:
            print(f"Loss function choice not recognised, exiting.")
            sys.exit(1)

        # Create our optimiser, if using reciprocal triplet loss, don't have a momentum component
        if "Reciprocal" in args.loss_function:
            optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimiser = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

        return data_loader, model, loss_fn, optimiser

    # Save a checkpoint as the current state of training
    def saveCheckpoint(self, epoch, model, optimiser, description):
        # Construct a state dictionary for the training's current state
        state = {   'epoch': epoch+1,
                    'model_state': model.state_dict(),
                    'optimizer_state' : optimiser.state_dict()  }

        # Construct the full path for where to save this
        self.checkpoint_path = os.path.join(self.args.fold_out_path, f"{description}_model_state.pkl")

        # And save actually it
        torch.save(state, self.checkpoint_path)

    # Save training logs to file
    def saveLogs(self):
        # Save this data to file for plotting graphs, etc.
        np.savez(   self.log_path, 
                    loss_steps=self.loss_steps,
                    losses_mean=self.losses_mean,
                    losses_softmax=self.losses_softmax,
                    losses_triplet=self.losses_triplet,
                    accuracy_steps=self.accuracy_steps,
                    accuracies=self.accuracies    )

    # Log information 
    def logTrainInfo(self, epoch, step, loss_mean, loss_triplet=None, loss_softmax=None):
        # Add to our arrays
        self.loss_steps.append(step)
        self.losses_mean.append(loss_mean)
        if loss_triplet != None: self.losses_triplet.append(loss_triplet)
        if loss_softmax != None: self.losses_softmax.append(loss_softmax)
        
        # Construct a message and print it to the console
        log_message = f"Epoch [{epoch+1}/{self.args.num_epochs}] Global step: {step} | loss_mean: {loss_mean:.5f}"
        if loss_triplet != None: log_message += f", loss_triplet: {loss_triplet:.5f}"
        if loss_softmax != None: log_message += f", loss_softmax: {loss_softmax:.5f}"
        print(log_message)

        # Save this new data to file
        self.saveLogs()

    # Evaluate the current model state, calls test.py in a subprocess and saves the results to file
    def test(self, step):
        # Construct subprocess call string
        run_str  = f"python test.py"
        run_str += f" --model_path={self.checkpoint_path}"  # Saved model weights to use
        run_str += f" --dataset={self.args.dataset}"        # Which dataset to use
        run_str += f" --batch_size={self.args.batch_size}"  # Batch size to use when inferring
        run_str += f" --embedding_size={self.args.embedding_size}"  # Embedding dimensionality
        run_str += f" --current_fold={self.args.current_fold}"  # The current fold number
        run_str += f" --save_path={self.args.fold_out_path}"    # Where to store the embeddings
        run_str += f" --img_type={self.args.img_type}"      # Which image type we're using
        run_str += f" --exclude_difficult={self.args.exclude_difficult}" # Exclude difficult
        if "Softmax" in self.args.loss_function: run_str += f" --softmax_enabled=1" # Is softmax used

        # Let's run the command, decode and save the result
        accuracy = subprocess.check_output([run_str], shell=True)

        # It's the first accuracy we're interested in
        accuracy = float(accuracy.decode('utf-8').split("accuracy=")[1].split(";")[0])

        # Append global information
        self.accuracies.append(accuracy)
        self.accuracy_steps.append(step)

        # Report the accuracy
        print(f"Accuracy: {accuracy}%")

        # Save this accuracies to file
        self.saveLogs()

        return accuracy

    """
    Static methods
    """

    # Return the selected dataset based on text choice
    @staticmethod
    def selectDataset(args, split):
         # Load the selected dataset
        if args.dataset == "OpenSetCows2020":
            dataset = OpenSetCows2019(  args.unknown_ratio, 
                                        args.repeat_num, 
                                        split=split, 
                                        transform=True,
                                        combine=True,
                                        suppress_info=False )
        elif args.dataset == "RGBDCows2020":
            dataset = RGBDCows2020( fold=args.current_fold, 
                                    split=split, 
                                    img_type=args.img_type, 
                                    retrieval_mode="triplet",
                                    transform=True,
                                    suppress_info=False,
                                    exclude_difficult=args.exclude_difficult,
                                  )
        else:
            print(f"Dataset choice: {args.dataset} not recognised, exiting.")
            sys.exit(1)

        return dataset
        
    # Render a confusion matrix for testing statistics
    @staticmethod
    def confusionMatrix():
        filepath = "D:\\Work\\ATI-Pilot-Project\\src\\Identifier\\MetricLearning\\output\\D_testing_stats.json"
        with open(filepath, 'r') as handle:
            stats = json.load(handle)

        cm = confusion_matrix(stats['ground_truth'], stats['predicted'], normalize='all')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.rainbow)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        # plt.savefig(os.path.join(args.out_path, f"{img_type}_confusion_matrix.pdf"))
        
# Main/entry function
if __name__ == '__main__':
    Utilities.confusionMatrix()
