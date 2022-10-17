 #!/usr/bin/env python

# Core libraries
import os
from easydict import EasyDict as edict

"""
This file is for managing ALL constants and configuration values
They're all nested per python class

To import the file:
from config import cfg

easydict usage:
https://pypi.python.org/pypi/easydict/
"""

# The main configuration variable / dictionary
cfg = edict()

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

"""
Generic constants & parameters
"""

cfg.GEN = edict()

# User name on Blue Pebble/Crystal (CHANGE THIS TO YOUR OWN USERNAME)
cfg.GEN.BLUE_USERNAME = "ca0513"

# The base directory for this repository (automatically set based on available directories)
if os.path.exists("D:\\Work\\ATI-Pilot-Project"): # Home windows machine
	cfg.GEN.BASE_DIR = "D:\\Work\\ATI-Pilot-Project"
if os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"): # Blue pebble/crystal
	cfg.GEN.BASE_DIR = f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"

# If true, just logs to console, false it will log to file by process ID
cfg.GEN.LOG_TO_CONSOLE = True

"""
Coordinator constants
"""

cfg.COR = edict()

# For a video being processed, how long to wait (in seconds) inbetween individual frames that need processing
cfg.COR.FRAME_SKIP_TIME = 1

# How long in seconds to wait for before checking again for an unprocessed file
cfg.COR.CHECK_WAIT_TIME = 1


"""
Database constants
"""

cfg.DAT = edict()

# Whether to just output the data to a simple CSV file, if False it goes to a mysql database
cfg.DAT.TO_CSV = True

# Where to find the CSV database
if os.path.exists("D:\\Work\\ATI-Pilot-Project"):	# Home windows machine
	cfg.DAT.CSV_PATH = "D:\\Work\\ATI-Pilot-Project\\src\\Database\\data\\csv"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"):	# Blue Pebble/Crystal
	cfg.DAT.CSV_PATH = f"/work/{cfg.GEN.BLUE_USERNAME}/ATI-workspace/database/csv"

# Name of the CSV file
cfg.DAT.CSV_FILE = "data_output.csv"

"""
FileSystem constants
"""

cfg.FSM = edict()

# Where to find the file system workspace
if os.path.exists("D:\\Work\\ATI-Pilot-Project"):	# Home windows machine
	cfg.FSM.WORKSPACE = "D:\\Work\\ATI-Pilot-Project\\workspace"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"):	# Blue Pebble/Crystal
	cfg.FSM.WORKSPACE = f"/work/{cfg.GEN.BLUE_USERNAME}/ATI-workspace/videos"

# The folder of videos where copying is in progress
cfg.FSM.FOLDER_1 = "01_copying_in_progress"

# The folder of videos where copying has finished and files are ready to be processed
cfg.FSM.FOLDER_2 = "02_to_be_processed"

# The folder of videos that are currently being processed
cfg.FSM.FOLDER_3 = "03_processing_in_progress"

# The folder of videos that have finished being processed
cfg.FSM.FOLDER_4 = "04_processed"

# Video extension to expect for camera files
cfg.FSM.VIDEO_EXTENSION = "avi"

"""
Detector constants
"""

# The base dictionary
cfg.DET = edict()

# Dictionary for selecting between the possible detection methods
cfg.DETECTORS = edict()
cfg.DETECTORS.ROTATED_RETINANET = 0
cfg.DETECTORS.FASTER_RCNN_DETECTRON = 1
cfg.DETECTORS.PYTORCH_YOLOV3 = 2
cfg.DETECTORS.YOLOv3_DARKNETWRAPPER = 3
cfg.DETECTORS.YOLO3_PY = 4

# The currently selected detection method
cfg.DET.DETECTION_METHOD = cfg.DETECTORS.ROTATED_RETINANET
# cfg.DET.DETECTION_METHOD = cfg.DETECTORS.FASTER_RCNN_DETECTRON
# cfg.DET.DETECTION_METHOD = cfg.DETECTORS.PYTORCH_YOLOV3
# cfg.DET.DETECTION_METHOD = cfg.DETECTORS.YOLOv3_DARKNETWRAPPER
# cfg.DET.DETECTION_METHOD = cfg.DETECTORS.YOLO3_PY

# Where to find RetinaNet weights
if os.path.exists("D:\\Work\\ATI-Pilot-Project"):	# Home windows machine
	cfg.DET.RETINA_WEIGHTS_DIR = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Weights\\retinaNet"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"):	# Blue Pebble/Crystal
	cfg.DET.RETINA_WEIGHTS_DIR = f"/work/{cfg.GEN.BLUE_USERNAME}/models/RGBDCows2020/weights/retinanet"

# Which weights file to load at that directory
cfg.DET.RETINA_WEIGHTS_FILE = "jing-70-percent-of-data_OPTIMISER_WEIGHTS_REMOVED.h5"

# The batch size to use when running inference
cfg.DET.BATCH_SIZE = 16

# The detection confidence threshold to reject a positive detection as a false positive
cfg.DET.CONF_THRESHOLD = 0.3

# The intersection over union (IoU) threshold for accepting a detection as a TP with some ground truth box
cfg.DET.IoU_THRESHOLD = 0.5

# The non-maximum suppression threshold
cfg.DET.NMS_THRESHOLD = 0.5

"""
Identifier constants
"""

# The base dictionary
cfg.ID = edict()

# The batch size to use
cfg.ID.BATCH_SIZE = 16

# Dictionary for selecting between the possible identification methods
cfg.IDENTIFIERS = edict()
cfg.IDENTIFIERS.METRIC_LEARNING = 0
cfg.IDENTIFIERS.CLOSED_SET = 1

# The current selected identification method
cfg.ID.ID_METHOD = cfg.IDENTIFIERS.METRIC_LEARNING
# cfg.ID.ID_METHOD = cfg.IDENTIFIERS.CLOSED_SET

# Where to find MetricLearning weights and embeddings
if os.path.exists("D:\\Work\\ATI-Pilot-Project"):	# Home windows machine
	cfg.ID.ML_WEIGHTS_DIR = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Weights\\MetricLearning\\fold_2"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project"):	# Blue Pebble/Crystal
	cfg.ID.ML_WEIGHTS_DIR = f"/work/{cfg.GEN.BLUE_USERNAME}/models/RGBDCows2020/weights/latent-space/fold_0"

# Which weights file to load
cfg.ID.ML_WEIGHTS_FILE = "best_model_state.pkl"

# Where to find pre-inferred embeddings for K-NN
cfg.ID.ML_EMBED_0 = "train_embeddings.npz"
cfg.ID.ML_EMBED_1 = "test_embeddings.npz"

# Batch size to use for inference
cfg.ID.BATCH_SIZE = 16

# k for K-NN and classifying the embedded space
cfg.ID.K = 5

# Image size for input into the resnet-50 powered embedding function
cfg.ID.IMG_SIZE = (224, 224)

"""
Dataset constants
"""

# The base dictionary
cfg.DATASET = edict()

# Where to find the RGBDCows2020 dataset
if os.path.exists("D:\\Work\\ATI-Pilot-Project"): # Home Windows machine
	cfg.DATASET.RGBDCOWS2020_LOC = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\RGBDCows2020\\Identification"
elif os.path.exists(f"/work/{cfg.GEN.BLUE_USERNAME}/datasets/RGBDCows2020/Identification"): # Blue Pebble
	cfg.DATASET.RGBDCOWS2020_LOC = f"/work/{cfg.GEN.BLUE_USERNAME}/datasets/RGBDCows2020/Identification"

# Where to find the OpenSetCows2020 dataset
if os.path.exists("/home/will"): # Linux machine
	cfg.DATASET.OPENSETCOWS2020_LOC = "/home/will/work/1-RA/src/Datasets/data/OpenSetCows2019"
elif os.path.exists(f"/home/{cfg.GEN.BLUE_USERNAME}"): # Blue pebble
	cfg.DATASET.OPENSETCOWS2020_LOC = f"/work/{cfg.GEN.BLUE_USERNAME}/datasets/OpenCows2020/identification"
elif os.path.exists(f"/mnt/storage/home/{cfg.GEN.BLUE_USERNAME}"): # Blue crystal
	cfg.DATASET.OPENSETCOWS2020_LOC = f"/mnt/storage/home/{cfg.GEN.BLUE_USERNAME}/ATI-Pilot-Project/src/Datasets/data/OpenSetCows2019"
elif os.path.exists("D:\\OneDrive - University of Bristol"): # Home windows machine
	cfg.DATASET.OPENSETCOWS2020_LOC = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images"
