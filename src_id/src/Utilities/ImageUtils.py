# Core libraries
import os
import sys
import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageTk

# My libraries
sys.path.append("../../")
# from src.Utilities.DataUtils import DataUtils

# For image augmentation
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap

# Seed the augmenter
ia.seed(random.randint(0, 30000))

# Sometime lambda function
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define image augmentation sequence
seq = iaa.Sequential(
	[
		# apply the following augmenters to most images
		iaa.Fliplr(0.5), # horizontally flip 50% of all images
		iaa.Flipud(0.2), # vertically flip 20% of all images
		# crop images by -5% to 10% of their height/width
		sometimes(iaa.CropAndPad(
			percent=(-0.05, 0.1),
			pad_mode=ia.ALL,
			pad_cval=(0, 255)
		)),
		sometimes(iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
			rotate=(0, 360), # rotate by -45 to +45 degrees
			shear=(-16, 16), # shear by -16 to +16 degrees
			order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
			cval=(0, 255), # if mode is constant, use a cval between 0 and 255
			mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
		)),
		# execute 0 to 5 of the following (less important) augmenters per image
		# don't execute all of them, as that would often be way too strong
		iaa.SomeOf((0, 5),
			[
				# sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
				iaa.OneOf([
					iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
					iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
					iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
				]),
				# iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
				# iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
				# search either for all edges or for directed edges,
				# blend the result with the original image using a blobby mask
				# iaa.SimplexNoiseAlpha(iaa.OneOf([
				# 	iaa.EdgeDetect(alpha=(0.5, 1.0)),
				# 	iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
				# ])),
				iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
				# iaa.OneOf([
				# 	iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
				# 	iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
				# ]),
				# iaa.Invert(0.05, per_channel=True), # invert color channels
				# iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
				# iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
				# either change the brightness of the whole image (sometimes
				# per channel) or change the brightness of subareas
				# iaa.OneOf([
				# 	iaa.Multiply((0.5, 1.5), per_channel=0.5),
				# 	iaa.FrequencyNoiseAlpha(
				# 		exponent=(-4, 0),
				# 		first=iaa.Multiply((0.5, 1.5), per_channel=True),
				# 		second=iaa.ContrastNormalization((0.5, 2.0))
				# 	)
				# ]),
				# iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
				# iaa.Grayscale(alpha=(0.0, 1.0)),
				# sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
				# sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
				sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
			],
			random_order=True
		)
	],
	random_order=True
)

class ImageUtils:
	"""
	Class for storing static methods to do with image and video utilities
	"""

	@staticmethod
	def retrieveVideoProperties(video):
		""" Retrieve video properties for some video """

		# Carry out differently based on the OpenCV version
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		if int(major_ver) >= 3:	# OpenCV 3
			length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			fps = int(video.get(cv2.CAP_PROP_FPS))
			w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		else:	# OpenCV 2.4
			# Get the number of frames and FPS of the input video as well as the resolution
			length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
			fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
			w = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
			h = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

		return w, h, fps, length
	
	@staticmethod
	def augmentImage(image, objects, display=False):
		""" 
		Randomly augment an image according to the possiblities defined at the top of this file
		"""

		# Express bounding boxes in the imgaug format
		boxes = [ia.BoundingBox(x1=x['x1'], y1=x['y1'], x2=x['x2'], y2=x['y2'], label="cow") \
						for x in objects]
		bboxes = ia.BoundingBoxesOnImage(boxes, shape=image.shape)

		# Get the augmentation sequence
		seq_det = seq.to_deterministic()

		# Augment the image and corresponding bounding boxes
		image_aug = seq_det.augment_images([image])[0]
		bbs_aug = seq_det.augment_bounding_boxes([bboxes])[0]

		# Deal with boxes now outside the image or partially out of the image
		bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

		# Display the augmentation and corresponding boxes if asked to
		if display:
			image_before = bboxes.draw_on_image(image, thickness=2)
			image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])
			print(f"Before shape: {image_before.shape}")
			print(f"After shape: {image_after.shape}")
			cv2.imshow("before", image_before)
			cv2.imshow("after", image_after)
			cv2.waitKey(0)

		return image_aug, bbs_aug

	# Load an image into memory and resize it as required and padding with black
	@staticmethod
	def loadImageAtSize(img_path, size):
		# Make sure some file exists
		assert os.path.exists(img_path)

		# Load the image
		img = cv2.imread(img_path)

		# Resize it
		new_img = ImageUtils.resizeWithPadding(img, size)

		return new_img

	@staticmethod
	def resizeWithPadding(img, size):
		# Keep the original image size
		old_size = img.shape

		# Compute resizing ratio
		ratio = float(size[0])/max(old_size)
		new_size = tuple([int(x*ratio) for x in old_size])

		# Actually resize it
		img = cv2.resize(img, (new_size[1], new_size[0]))

		# Calculate where to position the image
		x_off = int(abs(img.shape[1] - size[1]) / 2)
		y_off = int(abs(img.shape[0] - size[0]) / 2)

		# Create a new image with the required dimensions
		new_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

		# Paste the resized image into the centre of this one
		new_img[y_off:y_off+img.shape[0], x_off:x_off+img.shape[1], :] = img

		return new_img

	# Transform a list of images from numpy into PyTorch form
	@staticmethod
	def npToTorch(images, return_tensor=False, resize=None):
		# Do we want to return a tensor or tuple of images?
		if not return_tensor:
			# List of converted images
			conv_imgs = []

		else:
			# We need to resize images to a consistent shape throughout the batch for this to work
			assert resize is not None

			# Numpy array of converted images
			np_images = np.zeros((len(images), 3, resize[0], resize[1]))

		# Iterate throught the list of images
		for i, img in enumerate(images):
			# Copy the image
			converted = img.copy()

			# Resize it with padding if needed
			if resize is not None:
				converted = ImageUtils.resizeWithPadding(converted, resize)

			# Transform from HWC -> CWH (PyTorch)
			converted = converted.transpose(2, 0, 1)

			# Do we want to return a tensor or tuple of images?
			if not return_tensor:
				# Convert to PyTorch
				converted = torch.from_numpy(converted).float()

				# Add it to the list
				conv_imgs.append(converted)
			else:
				# Paste the image into the larger numpy array
				np_images[i,:,:,:] = converted

		# Return the converted images as a tuple
		if not return_tensor: 
			return tuple(conv_imgs)

		# Return a tensor (batch of images)
		else:
			return torch.from_numpy(np_images).float()

	# Proportionally resize the image to a maximum of max_x or max_y
	@staticmethod
	def proportionallyResizeImageToMax(image, max_x, max_y):
		# Get the image shape
		width = image.shape[1]
		height = image.shape[0]

		# Resize the image
		if width > height:
			scale_factor = max_x / float(width)
			new_h = int(scale_factor * height)
			return cv2.resize(image, (max_x, new_h))
		else:
			scale_factor = max_y / float(height)
			new_w = int(scale_factor * width)
			return cv2.resize(image, (new_w, max_y))

	# Converts a OpenCV/numpy image array to the tk Photo format via PIL/Pillow
	@staticmethod
	def convertImage(np_image):
		image = Image.fromarray(np_image)
		image = ImageTk.PhotoImage(image)
		return image

	# Open and display imagery from a RTSP stream
	@staticmethod
	def openDisplayRTSPStream():
		stream_address = "rtsp://User:Password1@195.224.61.26:561/Streaming/Channels/101/"
		video_cap = cv2.VideoCapture(stream_address)

		w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		fps = video_cap.get(cv2.CAP_PROP_FPS)

		print(f"{w}x{h}@{fps}fps")

		while True:
			_, frame = video_cap.read()
			frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
			cv2.imshow("Video stream", frame)
			cv2.waitKey(1)

	# Open and display a video
	@staticmethod
	def playVideo(video_filepath):
		# Open the video file
		cap = cv2.VideoCapture(video_filepath)

		# Get the framerate
		fps = int(cap.get(cv2.CAP_PROP_FPS))

		# Loop until there are no more frames
		while cap.isOpened():
			ret, frame = cap.read()

			cv2.imshow(f"{os.path.basename(video_filepath)}", frame)
			cv2.waitKey(fps)

		cap.release()
		cv2.destroyAllWindows()

	# Given an image and rotated rectangle, extract the subimage from that rotated rectangle
	@staticmethod
	def extractRotatedSubImage(image, r_rect, visualise=False):
		# Convert the rotated rectangle to four pixel coordinates
		((x1,y1),(x2,y2),(x3,y3),(x4,y4)) = ImageUtils.rotatedRectToPixels(r_rect)

		# If there isn't a head centre/direction given, infer it from the top of the box
		if 'head_cx' not in r_rect.keys() and 'head_cy' not in r_rect.keys():
			r_rect['head_cx'] = int((x1+x2)/2)
			r_rect['head_cy'] = int((y1+y2)/2)

		# Visualise the lines, cow centre, head centre, top left corner
		if visualise:
			cv2.circle(image, (int(r_rect['cx']), int(r_rect['cy'])), 5, (255,0,0), 5)
			cv2.circle(image, (int(r_rect['head_cx']), int(r_rect['head_cy'])), 5, (0,0,255), 5)
			cv2.circle(image, (x1, y1), 5, (0,255,0),5)
			cv2.line(image, (x1,y1), (x2,y2), (255,0,0))
			cv2.line(image, (x2,y2), (x4,y4), (255,0,0))
			cv2.line(image, (x4,y4), (x3,y3), (255,0,0))
			cv2.line(image, (x3,y3), (x1,y1), (255,0,0))
			cv2.imshow('Image', image)

		# Create a numpy array with these four points
		contours = np.array([
								[[x1, y1]],
								[[x2, y2]],
								[[x3, y3]],
								[[x4, y4]]
							])

		# Find the area that bounds those points in box form
		rect = cv2.minAreaRect(contours)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		# What are the dimensions of this box
		width = int(rect[1][0])
		height = int(rect[1][1])

		# Create the standardised point sets
		src_pts = box.astype("float32")
		dst_pts = np.array([[0, height-1],
							[0, 0],
							[width-1, 0],
							[width-1, height-1]], dtype="float32")

		# Find the transformation (rotation) between these sets of points
		M = cv2.getPerspectiveTransform(src_pts, dst_pts)

		# Warp the image according to this
		cropped = cv2.warpPerspective(image, M, (width, height))

		# Transform cow and head points according to the transformation
		t_centre = np.matmul(M, [[r_rect['cx']], [r_rect['cy']], [1]])
		t_head = np.matmul(M, [[r_rect['head_cx']], [r_rect['head_cy']], [1]])

		# Visualise if we're supposed to
		if visualise: cv2.imshow('pre-cropped', cropped)

		# Rotate the image by 90 degrees so the cow is always horizontal
		h, w = cropped.shape[:2]
		if h > w:
			# Rotate in varying amounts of 90 degrees depending on whether the
			# head centre is above or below the cow centre
			if t_head[1][0] > t_centre[1][0]: cropped = np.rot90(cropped)
			else: cropped = np.rot90(cropped, k=3)
		# The image is horizontal, but may need to be rotated by 180 so that the
		# cow always faces to the right
		elif t_head[0][0] < t_centre[0][0]: cropped = np.rot90(cropped, k=2)

		# Visualise if we're supposed to
		if visualise: 
			cv2.imshow('Cropped', cropped)
			cv2.waitKey(0)

		return cropped

	# Given a rotated rectangle (cx, cy, w, h, angle), convert it to a set of integer pixel coordinates
	@staticmethod
	def rotatedRectToPixels(r_rect):
		# Get the centre and width height
		cx = r_rect['cx']
		cy = r_rect['cy']
		w = r_rect['w']
		h = r_rect['h']

		# Get the angle
		ang = r_rect['angle']

		# Find the hypotenuse (the same for all four corners)
		l = math.sqrt(pow(w/2, 2) + pow(h/2, 2))

		# Compute the different angles for the sides of the box
		a1 = ang + math.atan(h / float(w))
		a2 = ang - math.atan(h / float(w))

		# Compute the rotated points
		x1 = int(cx - l * math.cos(a1))
		y1 = int(cy - l * math.sin(a1))

		x2 = int(cx - l * math.cos(a2))
		y2 = int(cy - l * math.sin(a2))

		x3 = int(cx + l * math.cos(a2))
		y3 = int(cy + l * math.sin(a2))

		x4 = int(cx + l * math.cos(a1))
		y4 = int(cy + l * math.sin(a1))

		return ((x1,y1),(x2,y2),(x3,y3),(x4,y4))

	# Crop an image from the centre
	@staticmethod
	def centreCrop(img, amount):
		""" 
		Centre crop 

		Extended description of function. 

		Parameters: 
		arg1 (int): Description of arg1 

		Returns: 
		int: Description of return value 

		"""

		# Work out the amounts to crop by
		h, w = img.shape[:2]
		crop_w = int(amount * w)
		crop_h = int(amount * h)

		# Work out starting positions
		startx = w//2 - (crop_w//2)
		starty = h//2 - (crop_h//2)

		# Crop
		return img[starty:starty+crop_h, startx:startx+crop_w] 

# Entry method/unit testing
if __name__ == '__main__':
	# Test opening a RTSP stream
	# ImageUtils.openDisplayRTSPStream()

	# Test the ability to open and play a video in opencv
	path = "C:\\Users\\ca051\\Downloads\\192.168.1.64_01_2020042316591478.mp4"
	ImageUtils.playVideo(path)