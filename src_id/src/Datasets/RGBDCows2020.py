#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")
import cv2
import json
import pickle
import shutil
import random
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict

# Image interpretation
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# DL Stuff
import torch
import torchvision
from torch.utils import data

# My libraries
from config import cfg
from Utilities.DataUtils import DataUtils
from Utilities.ImageUtils import ImageUtils


class RGBDCows2020(data.Dataset):
	"""
	This Class manages everything to do with the RGBDCows2020 dataset and interfacing with it in PyTorch
	"""

	def __init__(	self,
					split_mode="trainvalidtest",
					fold=0,
					num_training_days=-1,
					num_testing_days=-1,
					split="train",				# train, test or valid
					img_type="RGB",				# D, RGB or RGBD
					retrieval_mode="single",	# Retrieve triplets or single images
					depth_type="normal",		# What type of depth image do we want (normal or binarised)
					img_size=(224, 224),		# Resolution to get images at
					augment=False,				# Whether to randomly augment images (only works for training)
					transform=False,			# Transform images to PyTorch form
					suppress_info=False,		# Suppress printing some info about this dataset
					exclude_difficult=False 	# Whether to exclude difficult categories
				):
		""" Class constructor

		Args:
			split_mode (str, optional): How are we splitting the dataset into training and testing? The default 
				"random" uses random splits from the entire data corpus and so may be heavily correlated. The second
				option "day" splits the dataset by day. If using "day" splits, then the number of days to use for
				training and testing need to be specified (num_training_days, num_testing_days)
			fold (int, optional): If we're in "random" split mode, which fold should we use to retrieve train
				test splits
			num_training_days (int, optional): If we're in "day" split mode, how many days should make up the 
				training set (there are 31 days total). This will choose num_training_days from the start of the
				ordered list of days
			num_testing_days (int, optional): As for num_training_days but will pick from the end of the list of
				days for the testing set.
			split (str, optional): Does this instance of the dataset want to retrieve training or test data?
		"""

		# Initialise superclass
		super(RGBDCows2020, self).__init__()

		# The root directory for the dataset itself
		self.__root = cfg.DATASET.RGBDCOWS2020_LOC

		# Which split mode are we in
		self.__split_mode = split_mode

		# Should we exclude a list of difficult examples from the dataset
		self.__exclude_difficult = exclude_difficult

		# The split we're after (e.g. train/test)
		self.__split = split

		# What image type to retrieve? (D, RGB or RGBD)
		self.__img_type = img_type

		# What retrieval mode are we after, single images or triplets (a, p, n)
		self.__retrieval_mode = retrieval_mode

		# Static list of difficult categories (e.g. all one colour with no markings) to be removed if requested
		if self.__exclude_difficult: self.__exclusion_cats = ["054", "069", "073", "173"]
		else: self.__exclusion_cats = []

		# If we're in triplet mode, remove animal 182 as it only has two instances and when split into train/valid/test
		# causes issues with finding positives and negatives
		if self.__retrieval_mode == "triplet":
			self.__exclusion_cats.append("182")

		# The fold we're currently using for train/test splits (if in random mode)
		self.__fold = str(fold)

		# Select and load the split file we're supposed to use
		if self.__split_mode == "random":
			self.__splits_filepath = os.path.join(self.__root, "random_10_fold_splits.json")
			assert os.path.exists(self.__splits_filepath)
			with open(self.__splits_filepath) as handle:
				self.__splits = json.load(handle)

		# We're splitting into train/valid/test (single fold for the time being)
		if self.__split_mode == "trainvalidtest":
			self.__splits_filepath = os.path.join(self.__root, "single_train_valid_test_splits.json")
			assert os.path.exists(self.__splits_filepath)
			with open(self.__splits_filepath) as handle:
				self.__splits = json.load(handle)

		# Split by day
		elif self.__split_mode == "day":
			self.__splits_filepath = os.path.join(self.__root, "day_splits.json")

			# Make sure the number of training or testing days have been specified
			if num_training_days < 0 and num_testing_days < 0:
				print("If using day splits, the number of training or testing days must be specified")
				sys.exit(1)
			# Load the file and make sure the specified numbers work out
			else:
				assert os.path.exists(self.__splits_filepath)
				with open(self.__splits_filepath) as handle:
					self.__splits = json.load(handle)

				if num_training_days + num_testing_days > len(self.__splits.keys()):
					print_str = f"The number of days specified ({num_training_days}+{num_testing_days})"
					print_str += f" exceed the total number of days {len(self.__splits.keys())}"
					print(print_str)
					sys.exit(1)

		# The folders containing RGB and depth folder datasets
		self.__RGB_dir = os.path.join(self.__root, "RGB")
		if depth_type == "normal": self.__D_dir = os.path.join(self.__root, "Depth")
		elif depth_type == "binarised": self.__D_dir = os.path.join(self.__root, "Depth_Binarised")
		assert os.path.exists(self.__RGB_dir)
		assert os.path.exists(self.__D_dir)

		# Retrieve the number of classes from both of these
		self.__RGB_folders = DataUtils.allFoldersAtDir(self.__RGB_dir, exclude_list=self.__exclusion_cats)
		self.__D_folders = DataUtils.allFoldersAtDir(self.__D_dir, exclude_list=self.__exclusion_cats)
		assert len(self.__RGB_folders) == len(self.__D_folders)
		self.__num_classes = len(self.__RGB_folders) + len(self.__exclusion_cats)

		# The image size to resize to
		self.__img_size = img_size

		# The complete dictionary of filepaths
		self.__files = {'train': [], 'valid':[], 'test': []}

		# The dictionary of filepaths sorted by ID
		self.__sorted = {'train': {}, 'valid':{}, 'test': {}}

		# Whether to transform images to PyTorch form
		self.__transform = transform

		# For PyTorch, which device to use, GPU or CPU?
		self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		"""
		Class setup
		"""

		# Iterate through each category
		for current_RGB, current_D in zip(self.__RGB_folders, self.__D_folders):
			# Make sure we're inspecting the same category
			raw_ID = os.path.basename(current_RGB)
			assert raw_ID == os.path.basename(current_D)

			# Find all the images within these folders
			RGB_paths = DataUtils.allFilesAtDirWithExt(current_RGB, ".jpg")
			D_paths = DataUtils.allFilesAtDirWithExt(current_D, ".jpg")
			assert len(RGB_paths) == len(D_paths)

			# There may be no validation files that get populated, have an empty array to add for this case
			valid_files = []

			# Populate the lists based off of which split mode we're in
			if self.__split_mode == "random":
				# Create the long lists of all training and testing filenames for this fold
				train_files = [{'class_ID': raw_ID, 'filename': x} for x in self.__splits[self.__fold][raw_ID]['train']]
				test_files = [{'class_ID': raw_ID, 'filename': x} for x in self.__splits[self.__fold][raw_ID]['test']]

				# Create the list of filenames sorted by category for this fold
				self.__sorted['train'][raw_ID] = list(self.__splits[self.__fold][raw_ID]['train'])
				self.__sorted['test'][raw_ID] = list(self.__splits[self.__fold][raw_ID]['test'])

			# We're using a train/valid/test file with a single fold
			if self.__split_mode == "trainvalidtest":
				# Create the long lists of all training and testing filenames for this fold
				train_files = [{'class_ID': raw_ID, 'filename': x} for x in self.__splits[raw_ID]['train']]
				valid_files = [{'class_ID': raw_ID, 'filename': x} for x in self.__splits[raw_ID]['valid']]
				test_files = [{'class_ID': raw_ID, 'filename': x} for x in self.__splits[raw_ID]['test']]

				# Create the list of filenames sorted by category for this fold
				self.__sorted['train'][raw_ID] = list(self.__splits[raw_ID]['train'])
				self.__sorted['valid'][raw_ID] = list(self.__splits[raw_ID]['valid'])
				self.__sorted['test'][raw_ID] = list(self.__splits[raw_ID]['test'])

			# We're in day split mode, populate differently
			elif self.__split_mode == "day":
				# Initialise the dict for this ID
				self.__sorted['train'][raw_ID] = []
				self.__sorted['test'][raw_ID] = []

				# Go through the specified training and testing days and add to the dictionaries
				train_files = []
				for i in range(num_training_days):
					current_day = sorted(self.__splits.keys())[i]
					files = self.__splits[current_day][raw_ID]
					files_dict = [{'class_ID': raw_ID, 'filename': file} for file in files]
					if len(files_dict) > 0: 
						train_files.extend(files_dict)
						self.__sorted['train'][raw_ID].extend(files)

				test_files = []
				for i in range(num_testing_days):
					current_day = sorted(self.__splits.keys())[-i-1]
					files = self.__splits[current_day][raw_ID]
					files_dict = [{'class_ID': raw_ID, 'filename': file} for file in files]
					if len(files_dict) > 0: 
						test_files.extend(files_dict)
						self.__sorted['test'][raw_ID].extend(files)

			# Populate the total list of files
			self.__files['train'].extend(train_files)
			self.__files['valid'].extend(valid_files)
			self.__files['test'].extend(test_files)

		# Print some info
		if not suppress_info: self.printStats(extended=False)

	"""
	Superclass overriding methods
	"""

	def __len__(self):
		"""
		Get the number of items for this dataset (depending on the split) 
		"""
		return len(self.__files[self.__split])

	def __getitem__(self, index):
		"""
		Superclass overriding retrieval method for a particular index
		"""

		# TODO: add augmentation possiblities

		# Extract the anchor filename and the class it belongs to
		anchor_filename = self.__files[self.__split][index]['filename']
		label_anchor = self.__files[self.__split][index]['class_ID']

		# Keep a copy of the original label
		label_anchor_orig = label_anchor

		# Convert to numpy form
		label_anchor = np.array([int(label_anchor)])

		# Construct the full path to the image based on which type we'd like to retrieve
		img_anchor = self.__fetchImage(label_anchor_orig, anchor_filename)

		# Transform the anchor image and corresponding label if we're supposed to
		if self.__transform:
			img_anchor = ImageUtils.npToTorch([img_anchor])[0]
			label_anchor = torch.from_numpy(label_anchor).long()

		# If we're in single image retrieval mode, stop here and return
		if self.__retrieval_mode == "single": 
			return img_anchor, label_anchor, anchor_filename

		# Otherwise we're in triplet mode
		elif self.__retrieval_mode == "triplet":
			# If we're retrieving the test or validation set, no need to bother finding a positive and negative
			if self.__split == "test" or self.__split == "valid": 
				return img_anchor, [], [], label_anchor, []

			# Load another random positive from this class
			img_pos = self.__retrievePositive(anchor_filename, label_anchor_orig)

			# Load a random negative from a different class
			img_neg, label_neg = self.__retrieveNegative(label_anchor_orig)

			# Convert label to numpy
			label_neg = np.array([int(label_neg)])

			# Transform positive and negative into PyTorch friendly form
			if self.__transform:
				# Convert the positive and negative images
				img_pos, img_neg = ImageUtils.npToTorch([img_pos, img_neg])

				# Convert the negative label
				label_neg = torch.from_numpy(label_neg).long()

			return img_anchor, img_pos, img_neg, label_anchor, label_neg

	"""
	Public methods
	"""

	def printStats(self, extended=False):
		"""Print statistics about this dataset"""
		print("__RGBDCows2020 Dataset___________________________________________________")
		print(f"Total number of categories: {self.__num_classes}")
		if self.__exclude_difficult: 
			print(f"Removed {len(self.__exclusion_cats)} difficult categories: {self.__exclusion_cats}")
		images_str = f"Found {len(self.__files['train'])} training images, "
		images_str += f"{len(self.__files['valid'])} validation images, "
		images_str += f"{len(self.__files['test'])} testing images."
		print(images_str)
		print(f"Current fold: {self.__fold}, current split: {self.__split}")
		print(f"Image type: {self.__img_type}, retrieval mode: {self.__retrieval_mode}, split mode: {self.__split_mode}")
		print("_________________________________________________________________________")

		# We want some extended information about this set
		if extended:
			assert self.__sorted['train'].keys() == self.__sorted['test'].keys()
			for k in self.__sorted['train'].keys():
				# Compute the number of images for this class
				total_images = len(self.__sorted['train'][k]) + len(self.__sorted['test'][k])

				# Highlight which classes have fewer instances than the number of folds
				if total_images < len(self.__splits.keys()):
					print(f"Class {k} has {total_images} images")

	# Visualise the images and labels one by one, transform needs to be false for
	# this function to work
	def visualise(self, shuffle=True):
		# Transform to PyTorch form needs to be false
		assert not self.__transform

		# PyTorch data loader
		trainloader = data.DataLoader(self, batch_size=1, shuffle=shuffle)

		# Visualise differently based on which retrieval mode we're in
		if self.__retrieval_mode == "single":
			for img, label, filename in trainloader:
				# Convert the label to the string format we're used to
				label_str = str(label.numpy()[0][0]).zfill(3)
				print(f"Class is: {label_str} for filename: {filename[0]}")

				# We just have a single image to display
				if self.__img_type == "RGB" or self.__img_type == "D":
					# Convert image from tensor to numpy
					disp_img = img[0].numpy().astype(np.uint8)

					# Display the image
					cv2.imshow("Anchor image", disp_img)

				# We have a RGB and depth image to display
				elif self.__img_type == "RGBD":
					# Convert image from tensor to numpy and extract RGB and depth components
					RGBD_img = img[0].numpy().astype(np.uint8)
					RGB_img = RGBD_img[:,:,:3]
					D_img = np.zeros(RGB_img.shape, dtype=np.uint8)
					D_img[:,:,0] = RGBD_img[:,:,3]
					D_img[:,:,1] = RGBD_img[:,:,3]
					D_img[:,:,2] = RGBD_img[:,:,3]

					# Concatenate the images vertically
					disp_img = np.concatenate((RGB_img, D_img), axis=0)
					cv2.imshow("Anchor RGB and D image", disp_img)

				# Wait for user keypress
				cv2.waitKey(0)

		# Visualise triplets
		elif self.__retrieval_mode == "triplet":
			for img, img_pos, img_neg, label, label_neg in trainloader:
				# Convert the labels to the string format
				str_lab = str(label.numpy()[0][0]).zfill(3)
				str_neg = str(label_neg.numpy()[0][0]).zfill(3)
				print(f"Anchor class: {str_lab}, negative: {str_neg}")

				# We just have RGB or D image to display
				if self.__img_type == "RGB" or self.__img_type == "D":
					# Convert images from tensor to numpy
					disp_anc = img[0].numpy()
					disp_pos = img_pos[0].numpy()
					disp_neg = img_neg[0].numpy()

				# We need to separate RGB and D
				elif self.__img_type == "RGBD":
					# Convert images from tensor to numpy
					RGBD_anc = img[0].numpy()
					RGBD_pos = img_pos[0].numpy()
					RGBD_neg = img_neg[0].numpy()

					# Get the RGB components
					RGB_anc = RGBD_anc[:,:,:3]
					RGB_pos = RGBD_pos[:,:,:3]
					RGB_neg = RGBD_neg[:,:,:3]

					# Create the D components
					D_anc = np.zeros(RGB_anc.shape, dtype=np.uint8)
					D_pos = np.zeros(RGB_pos.shape, dtype=np.uint8)
					D_neg = np.zeros(RGB_neg.shape, dtype=np.uint8)

					# Copy array slices across to each
					D_anc[:,:,0] = RGBD_anc[:,:,3] ; D_anc[:,:,1] = RGBD_anc[:,:,3] ; D_anc[:,:,2] = RGBD_anc[:,:,3]
					D_pos[:,:,0] = RGBD_pos[:,:,3] ; D_pos[:,:,1] = RGBD_pos[:,:,3] ; D_pos[:,:,2] = RGBD_pos[:,:,3]
					D_neg[:,:,0] = RGBD_neg[:,:,3] ; D_neg[:,:,1] = RGBD_neg[:,:,3] ; D_neg[:,:,2] = RGBD_neg[:,:,3]

					# Vertically concatenate RGB and D components
					disp_anc = np.concatenate((RGB_anc, D_anc), axis=0)
					disp_pos = np.concatenate((RGB_pos, D_pos), axis=0)
					disp_neg = np.concatenate((RGB_neg, D_neg), axis=0)

				# Concatenate the images into one image
				disp_img = np.concatenate((disp_anc, disp_pos, disp_neg), axis=1)
				cv2.imshow("Anchor, positive and negative images", disp_img)
				cv2.waitKey(0)

	def produceLIME(self, model_path, shuffle=True):
		""" 
		Produce LIME images

		Produce locally-interpretable model agnostic explanations to provide basis for qualitative assessment
		Based off of: https://github.com/marcotcr/lime and the tutorial for using LIME with PyTorch:
		https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

		Parameters: 
		arg1 (int): Description of arg1 

		Returns: 
		int: Description of return value 

		"""

		# Make sure there's something at the model path, load the weights if so
		assert os.path.exists(model_path)
		weights = torch.load(model_path)

		# Transform to PyTorch form needs to be False
		assert not self.__transform

		# We only want to be in single retrieval model
		assert self.__retrieval_mode == "single"

		# PyTorch data loader, loading one image at a time
		trainloader = data.DataLoader(self, batch_size=1, shuffle=shuffle)

		# Load the model from the supplied state dictionary file and put it on the selected device
		self.__model = torchvision.models.resnet50()
		self.__model.fc = torch.nn.Linear(self.__model.fc.in_features, self.getNumClasses())
		self.__model.load_state_dict(weights)
		self.__model.eval()
		self.__model.to(self.__device)

		# Create the image explainer
		explainer = lime_image.LimeImageExplainer()

		# Iterate through the dataset
		for image, label, _ in trainloader:
			# Give numpy array to explainer
			np_image = np.array(image[0,:,:,:])

			# Get explanation for this image
			explanation = explainer.explain_instance(np_image, self.__batchPredict, num_samples=1, top_labels=5, hide_color=0)

			# Visualise it
			# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
			# img_boundary1 = mark_boundaries(temp/255.0, mask)
			# plt.imshow(img_boundary1)

			# plt.show()

			temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=20, hide_rest=False)
			img_boundary2 = mark_boundaries(temp/255.0, mask)
			plt.imshow(img_boundary2)

			plt.show()

	"""
	(Effectively) private methods
	"""

	# Retrieve a random positive from this class that isn't the anchor
	def __retrievePositive(self, anchor_filename, label_anchor):
		# Copy the list of filenames for this category
		filenames = list(self.__sorted["train"][label_anchor])
		assert anchor_filename in filenames

		# Subtract the anchor path
		filenames.remove(anchor_filename)

		# Pick a random positive
		img_name = random.choice(filenames)

		# Load the image based on the image type we're after
		return self.__fetchImage(label_anchor, img_name)

	# Retrieve a random negative instance from the current split set
	def __retrieveNegative(self, label_anchor):
		# Copy the list of IDs
		IDs = list(self.__sorted["train"].keys())
		assert label_anchor in IDs

		# Subtract the anchor's ID
		IDs.remove(label_anchor)

		# Randomly select a category
		random_category = random.choice(IDs)

		# Randomly select a filename in that category
		img_name = random.choice(self.__sorted[self.__split][random_category])

		# Load the image based on the image type we're after
		return self.__fetchImage(random_category, img_name), random_category

	# Fetch the specified image based on its type, category and filename
	def __fetchImage(self, category, filename):
		# We want a 4-channel RGBD image
		if self.__img_type == "RGBD":
			# Construct the full paths to the RGB and Depth images
			RGB_path = os.path.join(self.__RGB_dir, category, filename)
			D_path = os.path.join(self.__D_dir, category, filename)

			# Load them both as RGB images
			RGB_img = ImageUtils.loadImageAtSize(RGB_path, self.__img_size)
			D_img = ImageUtils.loadImageAtSize(D_path, self.__img_size)

			# flatten D_img to single channel (should currently be 3 equal greyscale channels)
			assert np.array_equal(D_img[:,:,0], D_img[:,:,1])
			assert np.array_equal(D_img[:,:,1], D_img[:,:,2])
			D_img = D_img[:,:,0]

			# Combine into one 4-channel RGBD np array
			RGBD_img = np.concatenate((RGB_img, np.expand_dims(D_img, axis=2)), axis=2)

			return RGBD_img

		# We just want a standard RGB image
		elif self.__img_type == "RGB":
			img_path = os.path.join(self.__RGB_dir, category, filename)
			return ImageUtils.loadImageAtSize(img_path, self.__img_size)

		# We want just the depth image
		elif self.__img_type == "D":
			img_path = os.path.join(self.__D_dir, category, filename)
			return ImageUtils.loadImageAtSize(img_path, self.__img_size)

	# Helper function for produceLIME function in predicting on a batch of images
	def __batchPredict(self, batch):
		# Convert to PyTorch
		batch = batch[0,:,:,:].transpose(2, 0, 1)
		batch = torch.from_numpy(batch).float()
		batch = batch[None,:,:,:]

		# Put the batch on the GPU
		batch = batch.to(self.__device)

		# Get outputs on this batch
		with torch.no_grad():
			logits = self.__model(batch)

		# Get normalised probabilities from softmax
		probs = torch.nn.functional.softmax(logits, dim=1)

		return probs.detach().cpu().numpy()

	"""
	Getters
	"""

	def getNumClasses(self):
		return self.__num_classes

	def getDatasetPath(self):
		return self.__root

	def getSplitsFilepath(self):
		return self.__splits_filepath

	"""
	Setters
	"""

	"""
	Static methods
	"""

	# Go through each coloured depth image in the folder dataset and create an equivalent new dataset with
	# greyscale images only
	@staticmethod
	def undoColourMap():
		# grey_values = np.arange(256, dtype=np.uint8)
		# colour_values = cv2.applyColorMap(grey_values, cv2.COLORMAP_JET).reshape(256,3)
		# colour_values = np.concatenate(([[0,0,0]], colour_values), axis=0)

		# self.__colour_map = np.zeros((256,256,256), dtype=np.uint8)
		
		# for i in tqdm(range(256), desc="B"):
		# 	for j in tqdm(range(256), desc="G"):
		# 		for k in range(256):
		# 			closest_idx = np.array([LA.norm(x+y+z) for (x,y,z) in np.abs(colour_values - [i,j,k])]).argmin()
		# 			if closest_idx >= 256: closest_idx = 255
		# 			# print(colour_values[closest_idx])
		# 			self.__colour_map[i][j][k] = closest_idx

		# with open(os.path.join(self.__root, "colour_map.npz"), 'wb') as handle:
		# 	np.save(handle, self.__colour_map)

		# print(f"Finished generating colour map")
		# sys.exit(0)

		# for i in range(colour_values.shape[0]):
		# 	B = colour_values[i,0]
		# 	G = colour_values[i,1]
		# 	R = colour_values[i,2]

		# 	self.__colour_map[B][G][R] = i

		with open(os.path.join(self.__root, "colour_map.npz"), 'rb') as handle:
			self.__colour_map = np.load(handle)

		# grey_values = np.arange(256, dtype=np.uint8)
		# colour_values = map(tuple, cv2.applyColorMap(grey_values, cv2.COLORMAP_JET).reshape(256,3))
		# self.__colour_dict = dict(zip(colour_values, grey_values))

		uncoloured_path = os.path.join(self.__root, "Depth-Uncoloured")

		depth_folders = DataUtils.allFoldersAtDir(self.__D_dir)

		for folder in tqdm(depth_folders, position=0, leave=True):
			current_category = os.path.basename(folder)

			new_folder_path = os.path.join(uncoloured_path, current_category)

			new_folder = os.makedirs(new_folder_path, exist_ok=True)

			folder_contents = DataUtils.allFilesAtDirWithExt(folder, ".jpg")
			for img_path in tqdm(folder_contents, position=0, leave=True):
				image_name = os.path.basename(img_path)
				coloured = cv2.imread(img_path)

				discoloured = np.zeros(coloured.shape[:2], dtype=np.uint8)
				for i in range(coloured.shape[0]):
					for j in range(coloured.shape[1]):
						# A colour map exists for this pixel value
						B = coloured[i,j,0]
						G = coloured[i,j,1]
						R = coloured[i,j,2]
						if self.__colour_map[B][G][R] >= 0:
							# Quantise the pixel
							discoloured[i,j] = self.__colour_map[B][G][R]
						# Find the closest match
						else:
							print("SHOULDNT BE HERE")
							input()
				assert len(discoloured.shape) == 2

				new_path = os.path.join(new_folder_path, image_name)
				cv2.imwrite(new_path, discoloured)

	# EXERCISE CAUTION USING THIS FUNCTION, IT IS IRREVERSIBLE
	# For detection labelling, go into each motion-triggered folder and delete the depth videos to save some space
	@staticmethod
	def deleteDepthVideos(input_dir):
		# Get all the folders at the provided directory
		all_folders = DataUtils.allFoldersAtDir(input_dir)

		# Go through each one
		for folder in tqdm(all_folders):
			# Find the video files at this location
			videos = DataUtils.allFilesAtDirWithExt(folder, "avi")

			# Identify the depth video
			depth_path = [x for x in videos if "Depth.avi" in x]
			assert len(depth_path) == 1
			depth_path = depth_path[0]

			# Delete the file
			os.remove(depth_path)

	# For detection labelling, go into each motion-triggered folder and extract frames every ith frame from the RGB
	# or depth videos
	# If skip_frames is -1, it just goes for one frame a second (according to the video's FPS)
	@staticmethod
	def extractFrames(input_dir, output_dir, RGB=True, skip_frames=-1):
		# Get all the folders at the provided directory
		all_folders = DataUtils.allFoldersAtDir(input_dir)

		# Unique identifier for each extracted image
		image_ctr = 1

		# Monitor progress
		progress = 0
		total = len(all_folders)

		# Go through each one
		for folder in all_folders:
			# Find the video files at this location
			videos = DataUtils.allFilesAtDirWithExt(folder, "avi")

			# We might be after the RGB or depth video
			if RGB:
				# Identify and select the RGB video
				vid_path = [x for x in videos if "RGB.avi" in x]
			else:
				# Identify and select the depth video
				vid_path = [x for x in videos if "Depth.avi" in x]

			# Make sure there's only one present in the folder
			assert len(vid_path) == 1
			vid_path = vid_path[0]

			# Open the video
			cap = cv2.VideoCapture(vid_path)

			# Get the fps
			fps = int(cap.get(cv2.CAP_PROP_FPS))

			# Update the number of frames to skip
			if skip_frames == -1: skip_frames = fps - 1

			# Loop until the video is over
			while cap.isOpened():
				# Grab a frame
				ret, frame = cap.read()

				if ret:
					# Create the complete path we're going to write to
					save_path = os.path.join(output_dir, f"image_{str(image_ctr).zfill(7)}_{os.path.basename(folder)}.jpg")

					# Save the frame
					cv2.imwrite(save_path, frame)

					# Increment the frame counter
					image_ctr += 1

					# Skip some frames
					for i in range(skip_frames): _, _ = cap.read()
				else:
					break

			# Update on our progress
			progress += 1
			percentage = (float(progress) / total) * 100
			print(f"{progress}/{total} = {percentage:.2f}% completed")

	# Split the entire list of images into individual folders containing blocks of images
	@staticmethod
	def splitIntoBlocks(input_dir, output_dir, block_size=100):
		# Get all the image filepaths at the input directory
		image_filepaths = DataUtils.allFilesAtDirWithExt(input_dir, ".jpg")

		# Folder counter
		block_ctr = 1

		# Split the list of image filepaths into blocks
		path_blocks = DataUtils.chunks(image_filepaths, block_size)

		# Monitor progress
		progress = 0
		total = len(path_blocks)

		# Iterate over each block
		for block in path_blocks:
			# Create a folder for this block
			folder = os.path.join(output_dir, f"block_{str(block_ctr).zfill(3)}")
			os.makedirs(folder)

			# Move all images to this new location
			for img in block: os.rename(img, os.path.join(folder, os.path.basename(img)))

			# Increment the counter
			block_ctr += 1

			# Update on our progress
			progress += 1
			percentage = (float(progress) / total) * 100
			print(f"{progress}/{total} = {percentage:.2f}% completed")

	# From block folders, collate images (RGB, depth) and labels into seperate folders
	# Only move images across if they have an associated label
	# Also retrieves corresponding depth images that have an associated label and RGB
	# Remap to a common unique ID system
	@staticmethod
	def collateBlocks(blocks_dir, depth_dir, output_dir):
		# Construct the output directories
		rgb_out_path = os.path.join(output_dir, "RGB")
		dep_out_path = os.path.join(output_dir, "Depth")
		lab_out_path = os.path.join(output_dir, "Labels")

		# If they don't exist, create them
		if not os.path.exists(rgb_out_path): os.mkdir(rgb_out_path)
		if not os.path.exists(dep_out_path): os.mkdir(dep_out_path)
		if not os.path.exists(lab_out_path): os.mkdir(lab_out_path)

		# Determine all the folders in the blocks directory
		block_folders = DataUtils.allFoldersAtDir(blocks_dir)

		# File ID counter
		file_ctr = 1

		# Iterate over every folder
		for folder in tqdm(block_folders):
			# Find every label file in this folder
			lab_paths = DataUtils.allFilesAtDirWithExt(folder, "xml")

			# Iterate over every label file
			for label in lab_paths:
				# Get the base filename minus the file extension
				basename = os.path.basename(label)[:-4]

				# Construct the RGB and depth filepaths
				rgb = os.path.join(folder, basename)+".jpg"
				dep = os.path.join(depth_dir, basename)+".jpg"

				# If on windows, replace single backslashes with a forward slash
				rgb = '/'.join(rgb.split('\\'))
				dep = '/'.join(dep.split('\\'))

				# For idle capture, replace the _image with _depth
				dep = dep.replace("_image.jpg", "_depth.jpg")

				# Create the output file string
				file_str = str(file_ctr).zfill(5)

				# Ensure these files actually exist before copying files across
				if os.path.isfile(rgb) and os.path.isfile(dep):
					shutil.copyfile(rgb, os.path.join(rgb_out_path, file_str)+".jpg")
					shutil.copyfile(dep, os.path.join(dep_out_path, file_str)+".jpg")
					shutil.copyfile(label, os.path.join(lab_out_path, file_str)+".xml")
					
					# Increment the file counter
					file_ctr += 1
				else: 
					if not os.path.isfile(rgb): print(f"No RGB image at: {rgb}")
					if not os.path.isfile(dep): print(f"No Depth image at: {dep}")

	# The depth and RGB images are not perfectly aligned, compute the Homography
	# between them from annotated points and apply this to the depth images
	@staticmethod
	def findApplyTransformation(dataset_dir, visualise=False):
		# Construct the dataset directories
		# hom_path = os.path.join(dataset_dir, "homography")
		# depth_path = os.path.join(dataset_dir, "Depth")
		# aligned_path = os.path.join(dataset_dir, "depth-aligned")

		hom_path = "D:\\Work\\Data\\RGBDCows2020\\homography"
		depth_path = "D:\\Work\\Data\\RGBDCows2020\\depth-temp"
		aligned_path = "D:\\Work\\Data\\RGBDCows2020\\identification\\depth-aligned"

		# Find the label files
		labels = DataUtils.allFilesAtDirWithExt(hom_path, ".xml")

		# Lists of points
		source_pts = []
		dest_pts = []

		# Store the depth image
		d_img = None

		# Iterate over each file
		for label in tqdm(labels):
			# Extract the points
			annotations = DataUtils.readXMLAnnotation(label)

			# Make a list
			points = [[int(x['x1']), int(x['y1'])] for x in annotations['objects']]

			# Add them to separate lists
			if "Depth" in label: 
				source_pts.extend(points)
				d_img = cv2.imread(annotations['image_path'])
			elif "RGB" in label: dest_pts.extend(points)

			# Visualise if we're supposed to
			if visualise:
				print(annotations['image_path'])
				img = cv2.imread(annotations['image_path'])
				for pt in points: cv2.circle(img, (pt[0], pt[1]), 5, (0,255,0), 5)
				cv2.imshow("Visualisation", img)
				cv2.waitKey(0)

		# Convert lists to a numpy matrix
		assert len(source_pts) == len(dest_pts)
		np_source_pts = np.array(source_pts, np.float32)
		np_dest_pts = np.array(dest_pts, np.float32)
		assert np_source_pts.shape == np_dest_pts.shape

		# Let's determine the transformation between the two
		T = cv2.getPerspectiveTransform(np_source_pts, np_dest_pts)

		# Transform depth images
		d_img = cv2.warpPerspective(d_img, T, (d_img.shape[1], d_img.shape[0]))

		# Visualise
		for pt in dest_pts: cv2.circle(d_img, (pt[0], pt[1]), 5, (0,255,0), 5)
		cv2.imshow("Visualisation", d_img)
		cv2.waitKey(0)

		# Apply the change to all depth images
		depth_images = DataUtils.allFilesAtDirWithExt(depth_path, ".jpg")
		for img in tqdm(depth_images):
			d_img = cv2.imread(img)
			d_img = cv2.warpPerspective(d_img, T, (d_img.shape[1], d_img.shape[0]))
			write_dir = os.path.join(aligned_path, os.path.basename(img))
			cv2.imwrite(write_dir, d_img)

	# Visualise RGB & depth images and overlay the bbox annotation


	# Goes through a folder full of labels and standardises the metadata
	@staticmethod
	def standardiseLabelMetadata(dataset_dir):
		# Construct directory to labels
		labels_path = os.path.join(dataset_dir, "Labels")

		# Find all the labels at that location
		label_files = DataUtils.allFilesAtDirWithExt(labels_path, ".xml")

		# Iterate through each one
		for label in tqdm(label_files):
			if "00001" in label: continue

			# Open the XML file
			tree = ET.parse(label)
			root = tree.getroot()

			# Get the parent annotation and remove the verified component
			del root.attrib['verified']

			# Work out and change the filename
			filename = os.path.basename(label)[:-4]
			root.find('filename').text = filename

			# Remove the folder tag
			root.remove(root.find('folder'))

			# Remove the path tag
			root.remove(root.find('path'))

			# Remove the segmented tag
			root.remove(root.find('segmented'))

			# Change the source tag
			root.find('source').find('database').text = "RGBDCows2020"

			# Go through each object and remove elements
			for obj in root.findall('object'):
				obj.remove(obj.find('pose'))
				obj.remove(obj.find('truncated'))
				obj.remove(obj.find('difficult'))

			# Write out the changes to file
			tree.write(label)

	# Retrieve full images from RoIs that were marked as difficult
	@staticmethod
	def retrieveDifficultImages(difficult_dir, blocks_dir, out_dir):
		# Find all difficult folders (IDs)
		difficult_folders = DataUtils.allFoldersAtDir(difficult_dir)

		# Find all block folders
		block_folders = DataUtils.allFoldersAtDir(blocks_dir)

		# Find all images within these blocks
		block_images = [DataUtils.allFilesAtDirWithExt(f, ".jpg") for f in block_folders]

		# Collapse into a 1D list
		block_images = [j for sub in block_images for j in sub]

		# Iterate through each one
		for folder in tqdm(difficult_folders):
			# Get the current folder ID
			ID = os.path.basename(folder)

			# Create a corresponding folder in the output directory
			if not os.path.exists(os.path.join(out_dir, ID)):
				os.makedirs(os.path.join(out_dir, ID))

			# Find each image in this folder
			query_images = DataUtils.allFilesAtDirWithExt(folder, ".jpg")

			# Iterate through every image
			for query_img in tqdm(query_images):
				# Get the raw filename, strip the extension and RoI counter, then
				# add the file extension
				filename = os.path.basename(query_img)[:-12]+".jpg"

				# Search for this filename in the list of block images
				for block_img in block_images:
					# If there is a match
					if filename in block_img:
						# Create the save to path
						save_path = os.path.join(out_dir, ID, filename)

						# Copy this image to the new location
						shutil.copy(block_img, save_path)

						# Skip the rest of the search for this query image
						break

	# For a folder of input images, separate them by day of capture in individuals
	# folders
	@staticmethod
	def separateImagesByDay(input_dir, output_dir):
		# Get the list of images
		image_fps = DataUtils.allFilesAtDirWithExt(input_dir, "jpg")

		# Two examples of possible naming conventions
		# 2020-02-11_12-27-53_image
		# image_0008030_2020-02-19_13-56-22

		# The list of dates we've encountered
		dates = []

		# Keep track of the number of duplicates
		duplicates = 0

		# Iterate over each one
		for image_path in tqdm(image_fps):
			# Strip the path to the filename
			filename = os.path.basename(image_path)

			# There are two image source with slightly different naming conventions
			# strip the prefix if it exists
			try:
				if filename.index("image_") == 0: filename = filename[14:]
			except Exception as e: pass

			# Split the string by underscores, the date is the first element
			datestring = filename.split("_")[0]

			# See whether this date has been seen before
			if datestring not in dates:
				# Add it to the list
				dates.append(datestring)

				# Create the directory
				folder_path = os.path.join(output_dir, datestring)
				assert not os.path.exists(folder_path)
				os.makedirs(folder_path)

			# Create the destination path
			destination = os.path.join(output_dir, datestring, os.path.basename(image_path))

			# Check whether this file exists already (images may have contained
			# multiple individuals)
			if os.path.exists(destination): 
				print(f"Duplicate file at destination: {destination}")
				print(f"Originates from: {image_path}")
				print()
				duplicates += 1
			else:
				shutil.copy(image_path, destination)

		print(f"Encountered {duplicates} duplicates for {len(image_fps)} images")

	@staticmethod
	def separateFolderDatasetByDay():
		"""
		For a folder dataset, generate a json file of filepaths for each category separated by day
		"""

		# Get the number of classes
		num_classes = RGBDCows2020().getNumClasses()

		# Find where RGB images from the dataset are located
		dataset_dir = RGBDCows2020().getDatasetPath()
		RGB_dir = os.path.join(dataset_dir, "RGB")

		# Load the folder dataset
		dataset = DataUtils.readFolderDatasetFilepathList(RGB_dir, full_path=False)

		# The dictionary we're trying to populate
		separated = {}

		# Keep count of the total number of images
		image_ctr = 0

		# Iterate through each category
		for category, filepaths in tqdm(dataset.items()):
			# Iterate through the list of filepaths
			for image_path in filepaths:
				# Strip the path to the filename
				filename = os.path.basename(image_path)

				# There are two image sources with slightly different naming conventions
				# strip the prefix if it exists
				try:
					if filename.index("image_") == 0: filename = filename[14:]
				except Exception as e: pass

				# Split the string by underscores, the date is the first element
				datestring = filename.split("_")[0]

				# Does this datestring already exist in our dictionary
				if datestring not in separated.keys():
					# Add it to our dictionary and populate the
					separated[datestring] = {str(cat).zfill(3): [] for cat in range(num_classes)}

				# Add this filepath to the dictionary
				separated[datestring][category].append(image_path)

				# Increment the counter
				image_ctr += 1

		# Convert to ordered dictionary sorted by key (the date)
		separated_sorted = OrderedDict(sorted(separated.items(), key=lambda t: t[0]))

		# Save the json file out
		save_path = os.path.join(dataset_dir, "day_splits.json")
		with open(save_path, 'w') as handle:
			json.dump(separated_sorted, handle, indent=2)

		# Print some stats about each day
		for day, day_data in separated_sorted.items():
			# Count the number of instances for this day
			day_total = 0

			# Count the number of categories that don't have any instances this day
			missing_count = 0

			# Iterate through this day
			for category, images in day_data.items():
				# Add to this day's total
				day_total += len(images)

				# Does this category have no instances
				if len(images) == 0: missing_count += 1

			# Calculate overall percentages
			percentage = (float(day_total) / image_ctr) * 100
			print_str = f"Date {day} had {percentage:.3}% of total data ({day_total}/{image_ctr} images)"
			print_str += f", {missing_count}/{num_classes} categories with no instances."
			print(print_str)

	"""
	Identification dataset static methods
	"""

	# For the folder dataset containing the identification dataset, create an identically
	# structured folder dataset containing the corresponding depth cattle RoIs
	@staticmethod
	def extractDepthRoIsForIDSet(RGB_dir, depth_dir, output_dir):
		# Load the folder dataset
		dataset = DataUtils.readFolderDatasetFilepathList(RGB_dir)

		# Iterate through each key
		for k in tqdm(dataset.keys()):
			# Create a corresponding folder in the output directory for this class
			depth_cls_dir = os.path.join(output_dir, k)
			os.makedirs(depth_cls_dir, exist_ok=True)

			# Iterate through each RGB image for this class
			for img_fp in dataset[k]:
				# Strip the filepath down
				image_filename = os.path.basename(img_fp)

				# Construct the filepath to the corresponding depth image
				depth_filepath = os.path.join(depth_dir, image_filename)

				# Make sure this file actually exists
				if os.path.exists(depth_filepath):
					# Create the filepath to the copy destination
					dest_fp = os.path.join(depth_cls_dir, image_filename)

					# Do the copying
					shutil.copy(depth_filepath, dest_fp)

				# Report there was a problem
				else:
					print(f"No corresponding depth image for: {image_filename}")

	@staticmethod
	def binariseDepthMaps(depth_dir, output_dir, threshold=115, visualise=False):
		# Load the folder dataset
		dataset = DataUtils.readFolderDatasetFilepathList(depth_dir)

		# Iterate through each key
		for k in tqdm(dataset.keys()):
			# Create a corresponding folder in the output directory for this class
			depth_cls_dir = os.path.join(output_dir, k)
			os.makedirs(depth_cls_dir, exist_ok=True)

			# Iterate through each Depth image for this class
			for img_fp in dataset[k]:
				# Load the image into memory
				d_img = cv2.imread(img_fp)

				# Get the base filename
				filename = os.path.basename(img_fp)

				# Threshold it
				_, binarised = cv2.threshold(d_img, threshold, 255, cv2.THRESH_BINARY)

				# Visualise
				if visualise:
					cv2.imshow("Depth image", d_img)
					cv2.imshow("Binarised", binarised)
					cv2.waitKey(0)

				# Construct the new filepath and save it
				dest_fp = os.path.join(depth_cls_dir, filename)
				cv2.imwrite(dest_fp, binarised)

# Entry method/unit testing method
if __name__ == '__main__':
	"""
	Static methods to do with annotation and manipulating the dataset for detection and ID
	purposes
	"""

	# Static methods for annotation
	# base_dir = "D:\\Work\\Data\\RGBDCows2020"
	# base_dir = "/mnt/storage/scratch/data/Wyndhurst/video_2020-03-19"

	# videos_path = os.path.join(base_dir, "TRIGERRED_CAPTURE")
	# frames_path = os.path.join(base_dir, "extracted_frames")
	# blocks_path = os.path.join(base_dir, "blocks")
	# depth_path = os.path.join(base_dir, "depth-temp")
	# output_path = os.path.join(base_dir, "Detection")
	# idle_path = os.path.join(base_dir, "IDLE_CAPTURE")

	# Extract RGB frames
	# RGBDCows2020.extractFrames(videos_path, frames_path, RGB=True)

	# Extract Depth frames
	# RGBDCows2020.extractFrames(videos_path, frames_path, RGB=False)

	# Split the images into blocks for labelling
	# RGBDCows2020.splitIntoBlocks(frames_path, blocks_path)

	# Collate labelled blocks
	# RGBDCows2020.collateBlocks(blocks_path, depth_path, output_path)

	# Visualise the detection dataset
	# RGBDCows2020.displayBboxes(output_path)

	# Let's find the fundamental transformation between depth and RGB images
	# RGBDCows2020.findApplyTransformation(os.path.join(base_dir, "identification\\depth"))

	# Let's standardise all the metadata from the label files
	# RGBDCows2020.standardiseLabelMetadata(output_path)

	# Extract full images to help identify difficult RoIs
	# RGBDCows2020.retrieveDifficultImages(	os.path.join(base_dir, "identification/difficult"),
	# 										blocks_path,
	# 										os.path.join(base_dir, "identification/difficult-full"))

	# Separate images by day
	# input_dir = os.path.join(base_dir, "identification/difficult-full")
	# output_dir = os.path.join(base_dir, "identification/difficult-full-separated")
	# RGBDCows2020.separateImagesByDay(input_dir, output_dir)

	# Generate json file of images separated by day
	# RGBDCows2020.separateFolderDatasetByDay()

	# Extract depth RoIs on the ID folder dataset
	# RGB_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\all\\RGB"
	# depth_dir = "D:\\Work\\Data\\RGBDCows2020\\identification\\tobelabelled"
	# output_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\all\\Depth"
	# RGBDCows2020.extractDepthRoIsForIDSet(RGB_dir, depth_dir, output_dir)

	# Produce binarised depth maps
	# depth_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\Depth"
	# out_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\Depth_Binarised"
	# RGBDCows2020.binariseDepthMaps(depth_dir, out_dir)

	"""
	Testing or otherwise for the actual class object, training, testing
	"""

	# Visualise the images and annotations via the PyTorch datasets object
	# (the same way they're accessed in a training or testing loop)
	# dataset = RGBDCows2020(	split_mode="day", 
	# 						num_training_days=1, 
	# 						num_testing_days=1, 
	# 						retrieval_mode="single", 
	# 						augment=False, 
	# 						img_type="RGB",
	# 						suppress_info=False	)
	dataset = RGBDCows2020(fold=0, retrieval_mode="single", img_type="D")
	dataset.visualise(shuffle=True)

	# Produce LIME explanations
	# fold = 0
	# img_type = "D"
	# base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Results\\ClosedSet"
	# model_path = os.path.join(base_dir, img_type, f"fold_{fold}_best_model_weights.pth") 
	# dataset = RGBDCows2020(fold=fold, split="train", img_type=img_type, transform=False)
	# dataset.produceLIME(model_path)
