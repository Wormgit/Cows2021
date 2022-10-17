# Core libraries
import os
import cv2
import sys
sys.path.append("../")
import math
import json
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom

# My libraries
from Utilities.ImageUtils import ImageUtils

class DataUtils:
	"""
	Class for storing static methods to do with data manipulation
	"""

	@staticmethod
	def chunks(lst, n):
		"""Yield successive n-sized chunks from lst."""
		for i in range(0, len(lst), n):
			yield lst[i:i + n]

	@staticmethod
	def extractRotatedRoIs(input_folder, output_folder, extract_depth=False, depth_path=None):
		"""
		Given a folder full of images and corresponding rotated annotation files, extract the rotated RoIs
		If extract depth is true, look into the given folder and extract RoIs there instead
		"""

		print(f"Extracting from: {input_folder}")

		# Make sure we have a path to the depth images
		if extract_depth: assert depth_path is not None

		# Get all the annotation files
		anno_fps = DataUtils.allFilesAtDirWithExt(input_folder, "xml")

		# Iterate over every annotation file
		for anno_fp in tqdm(anno_fps):
			# Extract the annotations
			anno = DataUtils.readRotatedXMLAnnotation(anno_fp)

			# Load the RGB or depth image
			if extract_depth:
				# Create the filepath to the depth image
				image_filename = anno['image_filename']

				# Some image filenames need a substring replacing
				image_filename = image_filename.replace("_image.", "_depth.")

				# From this, create the full path
				complete_imagepath = os.path.join(depth_path, image_filename)
			else:
				# Create the complete path to this current image
				complete_imagepath = os.path.join(input_folder, anno['image_filename'])

			# Load the image into memory
			image = cv2.imread(complete_imagepath)

			# Have a unique object counter
			obj_ctr = 1

			# Iterate over every object in the annotation file
			for obj in anno['objects']:
				# Extract the current object RoI
				roi = ImageUtils.extractRotatedSubImage(image, obj)

				# Create the RoI image name to save to file
				image_basename = anno['image_filename'][:-4]
				save_path = os.path.join(output_folder, f"{image_basename}_roi_{str(obj_ctr).zfill(3)}.jpg")

				# Actually save it
				cv2.imwrite(save_path, roi)

				# Increment the counter
				obj_ctr += 1

	@staticmethod
	def readRotatedXMLAnnotation(filepath):
		""" Read rotated annotation in VOC format into a dict """

		# Load the XML
		tree = ET.parse(filepath)

		# Which jpg file is this label pointing to
		image_fp = tree.find('filename').text + ".jpg"

		# Dictionary object for this annotation file
		annotation = {}

		# List of cow objects in this annotation file
		objects = []

		# The list of heads in this annotation file
		heads = []

		# First, find the head centres
		for obj in tree.findall('object'):
			# Only look at the head objects
			if obj.find('name').text == "head":
				head = {}

				# Get the head direction centre (just take the first point)
				# as a non-rotated bounding box
				try:
					head['cx'] = float(obj.find('bndbox').find('xmin').text)
					head['cy'] = float(obj.find('bndbox').find('ymin').text)
				except Exception as e1:
					# Failing that, try finding it as a rotated bbox
					try:
						head['cx'] = float(obj.find('robndbox').find('cx').text)
						head['cy'] = float(obj.find('robndbox').find('cy').text)
					except Exception as e2:
						print(f"Couldn't find head element in {filepath}")
						raise e2

				# Add it the list
				heads.append(head)

		# Loop over all the objects contained in the file
		for obj in tree.findall('object'):
			# Only look at cow objects
			if obj.find('name').text == "cow":
				# The cow dictionary object
				cow = {}

				# Get the values of the rotated bounding box
				try:
					cow['cx'] = float(obj.find('robndbox').find('cx').text)
					cow['cy'] = float(obj.find('robndbox').find('cy').text)
					cow['w'] = float(obj.find('robndbox').find('w').text)
					cow['h'] = float(obj.find('robndbox').find('h').text)
					cow['angle'] = float(obj.find('robndbox').find('angle').text)
				except Exception as e:
					print(f"Couldn't find cow robndbox element in {filepath}")
					raise e

				# Try and get a score (it may not exist)
				try:
					cow['score'] = float(obj.find('score').text)
				except Exception as e:
					pass
				
				# Normalise the angle
				# if cow['angle'] > math.pi: cow['angle'] -= math.pi
				# if cow['angle'] < -math.pi: cow['angle'] += math.pi

				# Always have the longest dimension as the height (may need swapping)
				# and adjust the angle for this
				if cow['w'] > cow['h']: 
					cow['h'], cow['w'] = cow['w'], cow['h']
					cow['angle'] -= math.pi/2
					if cow['angle'] < 0: cow['angle'] += 2*math.pi

				# Let's adjust the angle based on the direction of the head
				satisfied = False
				for head in heads:
					# Find whether this is the correct head direction dictionary
					if (cow['w']/2)**2 > (head['cx'] - cow['cx'])**2 + (head['cy'] - cow['cy'])**2:
						# Add info about the centre of the head direction
						cow['head_cx'] = head['cx']
						cow['head_cy'] = head['cy']

						# Find the angle of the head relative to the cow centre, add an offset to end up with
						# 0/pi being north
						head_ang = math.atan2((head['cy'] - cow['cy']), (head['cx'] - cow['cx']))
						head_ang = math.degrees(head_ang + math.pi/2)

						# Convert atan2 back to normal range from [0, 360]
						head_ang = (head_ang + 360) % 360
						head_ang = math.radians(head_ang)

						# Find the difference between the directions
						diff = abs(cow['angle'] - head_ang)

						# Determine the closest angle in math.pi/2 sized steps
						angles = np.asarray([0, math.pi/2, math.pi, 3*math.pi/2, 2*math.pi])
						idx = (np.abs(angles - diff)).argmin()
						closest = angles[idx]

						# Act based on this
						if closest == math.pi: cow['angle'] += math.pi

						# We can stop here
						satisfied = True
						break

				# We didn't find a head for this cow object, flag it to the user
				if not satisfied:
					print(f"Did not find corresponding head for file: {image_fp}")

				# Add this to the list of objects for this annotation
				objects.append(cow)

		# Add the objects and the filename this annotation refers to
		annotation['image_filename'] = image_fp
		annotation['objects'] = objects

		return annotation
	
	@staticmethod
	def readXMLAnnotation(filepath):
		""" Read VOC/faster-rcnn styled annotation file into a dict """

		# Load the XML
		tree = ET.parse(filepath)

		# Dictionary for this annotation
		annotation = {}

		# Which jpg file is this label pointing to
		try: annotation['filename'] = tree.find('filename').text
		except Exception as e: pass

		# Get the path
		try: annotation['image_path'] = tree.find('path').text
		except Exception as e: pass

		# List of objects in this annotation
		objects = []

		# Loop over all the objects
		for obj in tree.findall('object'):
			# Get the category of this object
			category = obj.find('name').text

			# Get the bounding box data in pixels
			x1 = int(obj.find('bndbox').find('xmin').text)
			y1 = int(obj.find('bndbox').find('ymin').text)
			x2 = int(obj.find('bndbox').find('xmax').text)
			y2 = int(obj.find('bndbox').find('ymax').text)

			# Clamp any negative values at zero
			if x1 < 0: x1 = 0
			if y1 < 0: y1 = 0
			if x2 < 0: x2 = 0
			if y2 < 0: y2 = 0

			# Swap components if necessary so that (x1,y1) is always top left
			# and (x2, y2) is always bottom right
			if x1 > x2: x1, x2 = x2, x1
			if y1 > y2: y1, y2 = y2, y1

			# Check for bad labels (it may be intentional)
			if x1 == x2 or y1 == y2:
				print(f"Box of zero size for file: {filepath}")

			# Add this all to the list of objects
			objects.append({'class': category, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

		# Store information about the image dimensions
		annotation['img_w'] = int(tree.find('size').find('width').text)
		annotation['img_h'] = int(tree.find('size').find('height').text)
		annotation['img_c'] = int(tree.find('size').find('depth').text)

		# Add the objects and the filename this annotation refers to
		annotation['objects'] = objects

		return annotation

	@staticmethod
	def writeXMLAnnotation(filepath, data_dict):
		""" Write VOC/faster-rcnn styled annotation file from a dict """

		# Create the root
		root = ET.Element("annotation")

		# Write the filename
		filename = ET.SubElement(root, "filename").text = data_dict['image_path']

		# Write some information about the image dimensions
		size = ET.SubElement(root, "size")
		width = ET.SubElement(size, "width").text = str(data_dict['img_w'])
		height = ET.SubElement(size, "height").text = str(data_dict['img_h'])
		depth = ET.SubElement(size, "depth").text = str(data_dict['img_c'])

		# Iterate over all the objects in this annotation
		for obj in data_dict['objects']:
			# Create the XML element
			object_e = ET.SubElement(root, "object")

			# Declare the type
			obj_type_e = ET.SubElement(object_e, "type").text = "bndbox"

			# Name it according to the class label
			name = ET.SubElement(object_e, "name").text = obj['class']

			# Create the bounding box object
			bndbox = ET.SubElement(object_e, "bndbox")
			xmin_e = ET.SubElement(bndbox, "xmin").text = str(obj['x1'])
			xmax_e = ET.SubElement(bndbox, "xmax").text = str(obj['x2'])
			ymin_e = ET.SubElement(bndbox, "ymin").text = str(obj['y1'])
			ymax_e = ET.SubElement(bndbox, "ymax").text = str(obj['y2'])

		# Prettify the XML to take up more than one line and be properly indented
		rough_string = ET.tostring(root, 'utf-8')
		reparsed = minidom.parseString(rough_string)
		tree = reparsed.toprettyxml(indent="    ")

		# And actually save it to file
		with open(filepath, "w") as text_file:
			text_file.write(tree)

	@staticmethod
	def readDarknetAnnotation(filepath, images_path=None):
		""" Read a darknet styled annotation file into a dict """

		# Create a list from each line (representing an annotation) in the text file
		lines = [line.rstrip() for line in open(filepath, "r")]

		# Dictionary for this annotation
		annotation = {}

		# List of objects in this annotation
		objects = []

		# The annotations and images may be in separate folders
		if images_path is not None:
			filename = os.path.basename(filepath)
			image_path = os.path.join(images_path, filename[:-4]+".jpg")
		else:
			# The complete path to the corresponding image for this annotation
			image_path = filepath[:-4] + ".jpg"

		# Load the image into memory
		img = cv2.imread(image_path)

		# If the image is none, the annotation file we're currently reading might
		# not actually be an annotation file and therefore won't have a corresponding
		# image
		if img is None:
			# Report that this is the case
			print(f"File at path is not an annotation file: {filepath}")
			return None

		# Otherwise, extract some information about the image
		annotation['img_w'] = img.shape[1]
		annotation['img_h'] = img.shape[0]
		annotation['img_c'] = img.shape[2]

		# Iterate through them
		for obj in lines:
			# Split the line into its components
			split = obj.split(" ")

			# The first element is the class ID (which is just 0 in every case)
			# so remove it
			split = split[1:]

			# Convert to pixels
			x1 = int((float(split[0]) - float(split[2])/2) * annotation['img_w'])
			y1 = int((float(split[1]) - float(split[3])/2) * annotation['img_h'])
			x2 = x1 + int(float(split[2]) * annotation['img_w'])
			y2 = y1 + int(float(split[3]) * annotation['img_h'])

			# Clamp any negative values at zero
			if x1 < 0: x1 = 0
			if y1 < 0: y1 = 0
			if x2 < 0: x2 = 0
			if y2 < 0: y2 = 0

			# Swap components if necessary so that (x1,y1) is always top left
			# and (x2, y2) is always bottom right
			if x1 > x2: x1, x2 = x2, x1
			if y1 > y2: y1, y2 = y2, y1

			# Check for bad labels
			if x1 == x2 or y1 == y2:
				print(f"Bad annotation for file: {filepath}")
			# This label is OK
			else:
				# Add this all to the list of objects
				objects.append({'class': "cow", 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

		# Add the objects and the filename this annotation refers to
		annotation['image_path'] = os.path.basename(image_path)
		annotation['objects'] = objects

		return annotation

	@staticmethod
	def writeDarknetAnnotation(filepath, data_dict):
		""" Write a darkney styled annotation from a dict """

		# Create the darknet label file for this instance
		text_file = open(filepath, "w")

		# Iterate over the objects in this annotation
		for obj in data_dict['objects']:
			# Convert pixel coordinates to darknet form
			dw = 1./data_dict['img_w']
			dh = 1./data_dict['img_h']
			x = (obj['x1'] + obj['x2'])/2.0 - 1
			y = (obj['y1'] + obj['y2'])/2.0 - 1
			w = obj['x2'] - obj['x1']
			h = obj['y2'] - obj['y1']
			x *= dw
			w *= dw
			y *= dh
			h *= dh

			# Write out a line
			text_file.write(f"0 {x} {y} {w} {h}\n")

		# We're finished
		text_file.close()

	@staticmethod
	def allFilesAtDirWithExt(directory, file_extension, full_path=True):
		""" 
		Create a sorted list of all files with a given extension at a given directory
		If full_path is true, it will return the complete path to that file
		"""

		# Make sure we're looking at a folder
		assert os.path.isdir(directory)

		# Gather the files inside
		if full_path:
			files = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]
		else:
			files = [x for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]

		return files

	@staticmethod
	def allFoldersAtDir(directory, full_path=True, exclude_list=None):
		""" Similarly, create a sorted list of all folders at a given directory """

		# Make sure we're looking at a folder
		if not os.path.isdir(directory): print(directory)
		assert os.path.isdir(directory)

		# Find all the folders
		if full_path:
			folders = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]
		else:
			folders = [x for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]

		# We might need to remove some folders
		if exclude_list is not None:
			temp_folders = folders.copy()
			folders = [x for x in folders if os.path.basename(x) not in exclude_list]

		return folders

	@staticmethod
	def XMLDatasetToFolder(input_labels_dir, input_images_dir, output_dir, train_test_split=-1.0):
		"""
		Converts a dataset from faster-rcnn format (xml plus image) to folder/image format
		e.g.
		Input:
		labels/	001.xml, 002.xml, ... (containing annotations: bbox + label for image)
		images/	001.jpg, 002.jpg, ... (the corresponding image)
		
		Output:
		class0/	001.jpg, 002.jpg, ... (Classwise RoI from input images)
		class1/	001.jpg, 002.jpg, ... ("	"	"	"	") 
		...
		In doing so, this method extracts classwise RoIs for each annotation in each input image
		"""

		# Get the list of labels and corresponding images
		label_files = DataUtils.allFilesAtDirWithExt(input_labels_dir, ".xml")
		image_files = DataUtils.allFilesAtDirWithExt(input_images_dir, ".jpg")

		# Make sure they're of equal size
		if len(image_files) != len(label_files):
			stripped_img = [os.path.basename(x)[:-4] for x in image_files]
			stripped_lab = [os.path.basename(x)[:-4] for x in label_files]
			difference = list(set(stripped_img) - set(stripped_lab))
			print(f"Images and labels don't match, difference={difference}")
			assert len(image_files) == len(label_files)

		# Dict where the category is the key and the value is a list of images/RoIs for that
		# category
		data_dict = {}

		# Loop over each label file
		print("Consolidating and loading annotations into memory")
		pbar = tqdm(total=len(label_files))
		for i in range(len(label_files)):
			# Read the annotation file
			annotation = DataUtils.readXMLAnnotation(label_files[i])

			# Load the image from this annotation, make sure its in our list
			image_filepath = os.path.join(input_images_dir, annotation['image_path'])
			assert image_filepath in image_files
			img = cv2.imread(image_filepath)

			# Create a list of subimages corresponding to the RoIs of objects extracted from
			# the XML file
			for obj in annotation['objects']:
				# Create a subimage from the object annotation
				RoI = img[obj['y1']:obj['y2'], obj['x1']:obj['x2']]

				# Check for a bad RoI
				if RoI.shape[1] == 0 or RoI.shape[0] == 0: 
					print(f"Bad RoI: ({obj['x1']}, {obj['y1']}, {obj['x2']}, {obj['y2']}), skipping")
				else:
					# If this is a newly discovered category, add the key first
					if obj['class'] not in data_dict.keys():
						data_dict[obj['class']] = [RoI]
					else:
						# Add it to the category/images dict
						data_dict[obj['class']].append(RoI)

			pbar.update()
		pbar.close()

		# Are we supposed to split the data randomly into a given ratio
		if train_test_split >= 0:
			pass

		# Write the dataset to file
		DataUtils.writeFolderDataset(data_dict, output_dir)

	@staticmethod
	def writeFolderDataset(data_dict, output_dir):
		""" Write folder dataset out to file """

		# Write out the data to file in individual folders
		print("Writing out files")
		for category, images_list in data_dict.items():
			# Create the complete category folder path and create it
			folder_path = os.path.join(output_dir, category)
			if not os.path.exists(folder_path): os.mkdir(folder_path)

			# For each image in the list for this category, save it out
			image_ctr = 1
			for img in images_list:
				filepath = os.path.join(folder_path, f"{str(image_ctr).zfill(6)}.jpg")
				cv2.imwrite(filepath, img)
				image_ctr += 1

	@staticmethod
	def readFolderDatasetImageList(dataset_dir):
		""" 
		Loads a folder/image dataset into memory (as created by XMLDatasetToFolder)
		Returns a dict of category: list of images
		"""

		# Get all the categories
		cat_dirs = DataUtils.allFoldersAtDir(dataset_dir)

		# The dataset dictionary
		dataset = {}

		# Loop over each category
		pbar = tqdm(total=len(cat_dirs))
		for folder in cat_dirs:
			# Extract the category name
			class_name = os.path.basename(folder)

			# Find all the images in this categories folder
			image_filepaths = DataUtils.allFilesAtDirWithExt(folder, ".jpg")

			# Instantiate this category with all the images loaded into numpy arrays
			dataset[class_name] = [cv2.imread(x) for x in image_filepaths]

			pbar.update()
		pbar.close()

		return dataset

	@staticmethod
	def readFolderDatasetFilepathList(dataset_dir, single_list=False, full_path=True):
		"""
		The same as "readFolderDatasetImageList", but a list of all complete image filepaths
		Images are NOT loaded into memory
		If single_list is true, it returns a list of filepaths for the entire set
		If false, the filepaths are organised into a dictionary by class
		If full_path is true, return the complete path, otherwise just the filename
		"""

		# Get all the categories
		cat_dirs = DataUtils.allFoldersAtDir(dataset_dir)

		# The dataset object
		if single_list: dataset = []
		else: dataset = {}

		# Loop over each category
		for folder in tqdm(cat_dirs):
			# Find all the images in this categories folder
			files = DataUtils.allFilesAtDirWithExt(folder, ".jpg", full_path=full_path)

			if single_list:
				dataset.extend(files)
			else:
				# Extract the category name
				class_name = os.path.basename(folder)

				# Create a new key with this list
				dataset[class_name] = files

		return dataset

	@staticmethod
	def splitFolderDatasetForKnownUknownCrossValidation(dataset_path, k, out_path, ratio=0.0):
		"""
		Split a folder dataset randomly into k folds for cross validation, this is primarily
		for use in randomly selecting known & unkown classes for open set testing
		The output is a .pkl file containing a dictionary with fold: list of known category
		IDs and a list of unknown category IDs
		
		If k == 1 and ratio is non-zero, split into two UNEVENLY sized chunks according to
		the specified ratio. The ratio defines the proportion of UNKNOWN classes
		"""

		# Load the entire dataset into memory
		dataset = DataUtils.readFolderDatasetFilepathList(dataset_path)

		# Copy the list of keys
		keys = list(dataset.keys())

		# Shuffle the list in place
		random.shuffle(keys)

		# Get the number of categories
		num_categories = len(keys)

		# Work out the chunk size
		n = math.ceil(num_categories / k)

		# Split the keys into these k evenly-sized chunks
		avg = num_categories / float(k)
		split = []
		last = 0.0
		while last < num_categories:
			split.append(keys[int(last):int(last+avg)])
			last += avg

		# We might want to split into two unevenly sized chunks
		if k == 1 and ratio > 0.0:
			unknown_size = int(ratio * num_categories)
			split = []
			split.append(keys[0:unknown_size])
			split.append(keys[unknown_size:])

		# The output dictionary
		folds_dict = {}

		# Populate the folds
		for i in range(k):
			# Create a dict for this fold
			fold_item = {}

			# unknown categories for this fold
			fold_item['unknown'] = list(split[i])

			# Known categories (all others)
			fold_item['known'] = list(set(keys) - set(split[i]))

			assert len(fold_item['unknown']) + len(fold_item['known']) == num_categories

			# Add it to the super dict
			folds_dict[i] = fold_item

		print(folds_dict)

		# Save the dictionary to file
		with open(out_path, 'wb') as handle:
			pickle.dump(folds_dict, handle)

	@staticmethod
	def splitFolderDatasetForCrossValidation(dataset_path, output_path, k):
		"""
		Similarly to the above but randomly split the imagesets for each class rather than
		the classes (i.e. general k-fold CV, not for known/unknown categories)
		Saves a json file with a dictionary
		"""

		# Load the entire folder dataset into memory
		dataset = DataUtils.readFolderDatasetFilepathList(dataset_path, full_path=False)

		# The dictionary of splits
		folds_dict = {i:{} for i in range(k)}

		# Iterate through each class
		for category in tqdm(dataset.keys()):
			# Copy the list of filepaths for this category
			filepaths = list(dataset[category])

			# Randomly shuffle the list in place
			random.shuffle(filepaths)

			# Work out the chunk size
			n = math.ceil(len(filepaths) / k)

			# Split the filepaths into these k evenly-sized chunks
			avg = len(filepaths) / float(k)
			split = []
			last = 0.0
			while last < len(filepaths):
				split.append(filepaths[int(last):int(last+avg)])
				last += avg

			# Populate the folds for this category
			for i in range(k):
				# Populate the dictionary for this fold and category
				folds_dict[i][category] = {'train': [], 'test': []}

				# Testing files
				folds_dict[i][category]['test'] = split[i]

				# Training files (all others)
				folds_dict[i][category]['train'] = list(set(filepaths) - set(split[i]))

			# Do some checks
			assert len(dataset[category]) == len(folds_dict[i][category]['test']) + len(folds_dict[i][category]['train'])

		# Save the dictionary to file
		with open(output_path, 'w') as handle:
			json.dump(folds_dict, handle, indent=1)

	@staticmethod
	def splitDatasetForCrossValidation(images_path, k, augmented_from=None):
		"""
		Split a folder full of images randomly into k folds for cross validation
		This is more geared towards a detection dataset with the provided folder just containing
		a load of images, with another containing bounding box annotations.
		"""

		# Load the list of image filepaths
		image_fps = DataUtils.allFilesAtDirWithExt(images_path, ".jpg", full_path=False)

		# The images have augmented instances from a particular filename upwards
		if augmented_from is not None:
			# Find the index of the first augmented sample
			aug_idx = image_fps.index(augmented_from)

			# Split the list into separate normal and synthetic lists using this
			normal = image_fps[:aug_idx]
			synthetic = image_fps[aug_idx:]
			assert len(image_fps) == len(normal) + len(synthetic)

			# Reassign the normal images with synthetics removed
			image_fps = normal

			# Shuffle the synthetic instances
			random.shuffle(synthetic)

			# Split these into k evenly sized chunks
			split_synth = np.array_split(np.array(synthetic), k)
			split_synth = [x.tolist() for x in split_synth]
		else: split_synth = None

		# Shuffle the list of images in place
		random.shuffle(image_fps)

		# Split this list into k evenly sized chunks using numpy, convert back to a normal list
		split = np.array_split(np.array(image_fps), k)
		split = [x.tolist() for x in split]

		# The dictionary containing the folds, key is the fold number
		folds_dict = {}

		# Convert this to a dict we can write it out to file
		fold_number = 0
		for i, test_files in enumerate(split):
			# Information for this fold
			fold = {}

			# Indicate the set of test images
			fold['test'] = test_files

			# Indicate the set of validation images
			valid_idx = (i+1)%len(split)
			valid_files = split[valid_idx]
			fold['valid'] = valid_files

			# Gather all the other files as training files
			temp = split.copy()
			temp.remove(test_files)
			temp.remove(valid_files)
			fold['train'] = [val for sublist in temp for val in sublist]

			# There might be some synthetic instances to add to the list of training images
			if split_synth is not None:
				fold['train'].extend(split_synth[i])

				# Run some checks
				for test_file in fold['test']:
					assert test_file not in synthetic
				for valid_file in fold['valid']:
					assert valid_file not in synthetic

			# Add this to the global folds dictionary
			folds_dict[fold_number] = fold

			# Increment the counter
			fold_number += 1

		# Save this out to a json file
		with open(f'{k}-fold-CV.json', 'w') as handle:
			json.dump(folds_dict, handle, indent=1)

	@staticmethod
	def splitCVFileForDarknet(prefix, CV_file, out_dir, k):
		""" Darknet just needs train/test .txt files per fold, separate a json file into this """

		# Load the JSON file
		with open(CV_file) as json_file:
			data = json.load(json_file)

		assert k == len(data.keys())

		# Iterate through each fold in turn
		for i in range(k):
			# Create the test file
			with open(os.path.join(out_dir, f"{i}-test.txt"), 'w') as handle:
				for line in data[str(i)]['test']:
					handle.write(os.path.join(prefix, line) + "\n")

			# Create the train file
			with open(os.path.join(out_dir, f"{i}-train.txt"), 'w') as handle:
				for line in data[str(i)]['train']:
					handle.write(os.path.join(prefix, line) + "\n")

			# If it exists, create a validation file too
			if 'valid' in data[str(i)].keys():
				with open(os.path.join(out_dir, f"{i}-valid.txt"), 'w') as handle:
					for line in data[str(i)]['valid']:
						handle.write(os.path.join(prefix, line) + "\n")

	@staticmethod
	def splitFolderDataset(dataset_path, num_test, train_valid_split=0.9):
		"""
		Split a folder dataset randomly into train/valid/test, saves to a json file
		Gets a constant n=num_test number of images for testing to solve testing imbalance
		"""

		# Load the entire dataset into memory
		dataset = DataUtils.readFolderDatasetFilepathList(dataset_path, full_path=False)

		# Train/test datasets
		split_dict = {}

		# Loop through each category
		for category in dataset.keys():
			# Copy the list of image instances
			images_copy = list(dataset[category])

			# Shuffle them in place
			random.shuffle(images_copy)

			# Create a dictionary for this category
			split_dict[category] = {}

			# If the category has fewer than num_test instances * 2, assign them all for training/validation
			if len(images_copy) < num_test*2: 
				print(f"Category: {category} only has {len(images_copy)} instances")

				remaining = images_copy
				split_dict[category]['test'] = []
			else:
				# Assign the first num_test images as testing images
				split_dict[category]['test'] = images_copy[:num_test]

				# Gather the remaining images
				remaining = images_copy[num_test:]

			# Compute a proportional split of train/valid
			train_num = math.floor(len(remaining)*train_valid_split)

			# And assign training and validation based off this
			split_dict[category]['train'] = remaining[:train_num]
			split_dict[category]['valid'] = remaining[train_num:]

			# Make a quick check
			assert len(images_copy) == len(split_dict[category]['train']) + len(split_dict[category]['valid']) + len(split_dict[category]['test'])

			# Print out some info
			print_str = f"Category: {category}, "
			print_str += f"#train: {len(split_dict[category]['train'])}, "
			print_str += f"#valid: {len(split_dict[category]['valid'])}, "
			print_str += f"#test: {len(split_dict[category]['test'])}"
			print(print_str)

		# Save out to a json file
		with open(os.path.join(dataset_path, "train_valid_test_splits.json"), 'w') as handle:
			json.dump(split_dict, handle, indent=1)

	@staticmethod
	def plotFolderDatasetDistribution(dataset_path):
		"""
		Plot a bar graph of the distribution of the number of instances across all categories
		of a folder dataset
		"""

		# Read the folder dataset
		dataset = DataUtils.readFolderDataset(dataset_path)

		# Get a list of the categories
		data_list = [[int(k), len(v)] for k, v in dataset.items()]

		# Convert to numpy
		data = np.asarray(data_list)

		plt.bar(data[:,0], data[:,1])
		plt.ylabel("Number of instances")
		plt.xlabel("Class ID")
		plt.show()

	@staticmethod
	def printPickleFile(filepath):
		""" Simply print the contents of a pickle file """

		with open(filepath, 'rb') as handle:
			contents = pickle.load(handle)

		print(contents)

	@staticmethod
	def printFolderDatasetStats(directory):
		""" For a folder dataset, report some stats about it """

		# Find all the folders (IDs) within this dataset (might include the unsure folder)
		folders = DataUtils.allFoldersAtDir(directory)

		# Determine the number of instances for each ID
		instances = [len(DataUtils.allFilesAtDirWithExt(x, ".jpg")) for x in folders]

		# Convert to numpy
		instances_np = np.array(instances)

		print(f"Number of IDs = {len(folders)}")
		print(f"Total number of instances = {np.sum(instances_np)}")
		print(f"Mean number of instances per ID = {np.mean(instances_np)}+-{np.std(instances_np)}")
		print(f"Min instances: {np.min(instances_np)}, max instances: {np.max(instances_np)}")

	@staticmethod
	def removeAugmented(directory, serial_number):
		""" Remove augmented samples from a validation file """

		# Find all files at this path
		txt_files = DataUtils.allFilesAtDirWithExt(directory, "txt")

		# Iterate through them
		for file in txt_files:
			# Only look at validation files
			if "-valid.txt" in file:
				# Load every line of the file into memory
				lines = open(file).read().splitlines()

				# List of lines we're holding onto for this file
				keep = []

				# Go through each one
				for line in lines:
					# Extract the number it refers to
					number = int(line.split("/")[-1][:-4])

					# Hold on to it if it doesn't surpass the threshold
					if number <= serial_number: keep.append(line)

				# Write out this new set of lines to a new file
				with open(file[:-4]+"-new.txt", "w") as handle:
					for new_line in keep:
						handle.write(new_line + "\n")

# Entry method/unit testing
if __name__ == '__main__':
	# Extract rotated RoIs from a folder
	# for i in range(1, 281):
	# 	input_folder = f"D:\\Work\\Data\\RGBDCows2020\\blocks\\block_{str(i).zfill(3)}"
	# 	output_folder = "D:\\Work\\Data\\RGBDCows2020\\identification\\tobelabelled"
	# 	depth_path = "D:\\Work\\Data\\RGBDCows2020\\identification\\depth-aligned"
	# 	DataUtils.extractRotatedRoIs(input_folder, output_folder, extract_depth=True, depth_path=depth_path)

	# Print folder dataset stats
	# input_folder = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\output"
	# input_folder = "D:\\Work\\Data\\RGBDCows2020\\identification\\difficult-full-separated"
	# input_folder = "D:\\Work\\ATI-Pilot-Project\\src\\Datasets\\data\\OpenSetCows2019\\split\\train"
	# DataUtils.printFolderDatasetStats(input_folder)

	# # Test getting a RoI from a rotated rectangle annotation
	# # anno = DataUtils.readRotatedXMLAnnotation('D:\\Work\\Data\\RGBDCows2020\\blocks\\block_249\\image_0024900_2020-03-10_12-46-38.xml')
	# anno = DataUtils.readRotatedXMLAnnotation('D:\\Work\\Data\\RGBDCows2020\\blocks\\block_012\\image_0001144_2020-02-11_12-31-19.xml')
	# # anno = DataUtils.readRotatedXMLAnnotation('C:/Users/ca051/Downloads/Capture.xml')
	# image = cv2.imread(anno['image_path'])
	# ImageUtils.extractRotatedSubImage(image, anno['objects'][1], visualise=True)

	# Convert a dataset
	# base_dir = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows/sources/Sion-Colin-Combined"
	# input_labels_dir = os.path.join(base_dir, "labels")
	# input_images_dir = os.path.join(base_dir, "images")
	# output_dir = os.path.join(base_dir, "folder_dataset")
	# train_test_split = -1
	# DataUtils.XMLDatasetToFolder(input_labels_dir, input_images_dir, output_dir, train_test_split)

	# Split a folder dataset into k folds for known/unknown (folds are stored in a pickle file)
	# k = 1
	# input_path = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/raw"
	# output_path = f"/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/{k}-folds.pkl"
	# DataUtils.splitFolderDatasetForKnownUknownCrossValidation(input_path, k, output_path, ratio=0.9)

	# dataset_path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\RGB"
	# output_path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\10-fold-CV-splits.json"
	# k = 10
	# DataUtils.splitFolderDatasetForCrossValidation(dataset_path, output_path, k)

	# Split a detection dataset into k folds of train, valid, test
	# images_path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\VOC"
	# # augmented_from = "003708.jpg"
	# augmented_from = None
	# DataUtils.splitDatasetForCrossValidation(images_path, 10, augmented_from=augmented_from)

	# Split a JSON cross validation file/dict into separate darknet-style train/test files
	# base_dir = "D:\\Work\\ATI-Pilot-Project\\src\\Utilities"
	# CV_file = os.path.join(base_dir, "10-fold-CV.json")
	# prefix = "/work/ca0513/datasets/CEADetection/images/"
	# DataUtils.splitCVFileForDarknet(prefix, CV_file, base_dir, 10)

	# Train/test split a folder dataset into distinct "train" & "test" folders randomly
	dataset_path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification\\RGB"
	num_test = 10
	DataUtils.splitFolderDataset(dataset_path, num_test, train_valid_split=0.9)

	# Plot the distiribution of raw instances for a folder dataset
	# dataset_path = "/home/will/work/1-RA/src/Datasets/data/CowID-PhD/raw"
	# DataUtils.plotFolderDatasetDistribution(dataset_path)

	# Just print the contents of a pickle file
	# filepath = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019/1-folds.pkl"
	# DataUtils.printPickleFile(filepath)
	
	# Remove augmented images from a validation text file given to darknet
	# path = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection\\splits"
	# path = "D:\\Work\\ATI-Pilot-Project\\src\\Utilities"
	# serial_number = 3707
	# DataUtils.removeAugmented(path, serial_number)

	# dataset = DataUtils.readFolderDatasetFilepathList("D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images\\train")
	# for k in sorted(dataset.keys()): 
		# print(f"/identification/images/train/{k} ({len(dataset[k])} images)")

	# See whether a folder full of darknet annotations contains any empties
	# base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\CEADetection"
	# annos_path = os.path.join(base_dir, "labels-darknet")
	# images_path = os.path.join(base_dir, "images")
	# for file in tqdm(DataUtils.allFilesAtDirWithExt(annos_path, ".txt")):
	# 	if len(DataUtils.readDarknetAnnotation(file, images_path=images_path).keys()) == 0:
	# 		print(file)