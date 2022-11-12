#!/usr/bin/env python
import argparse
import os, glob, warnings, sys
import keras, math
import keras.preprocessing.image
import tensorflow as tf

# a mistake is in BC4 so I added them
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Allow relative imports when being executed as script. in case other module call it
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    __package__ = "keras_retinanet.b_rotated"

import json
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
from ..utils.image import random_visual_effect_generator
from ..utils.gpu import setup_gpu
from ..preprocessing.coco import CocoGenerator

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None,nms_threshold = 0.5):
    """
    create a retinanet model with a given backbone.
        multi_gpu          : number. num_classes: to train 1 for cows weights to load
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.
    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : If multi_gpu=0, = model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None
    print(freeze_backbone)
    anchor_params = None # None : defaults will be used)
    num_anchors   = None
    if config and 'anchor_parameters' in config: # 没用上
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        print('\nGPU: using multiple gpu')
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        print('\nGPU: not using multiple gpu')
        model     = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, nms_threshold=nms_threshold)
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),  # dict 如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )
    return model, training_model, prediction_model

def create_callbacks(model, training_model, prediction_model, validation_generator, train_generator,args):
    """
    Args
        training_model.  prediction_model: validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    """
    callbacks = []
    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )

    if args.evaluation and validation_generator:
        from ..callbacks.coco import CocoEval,CocoEval2  # use prediction model for evaluation
        evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback, name_output=args.name_output) #threshold(score)=0.05 defult
        evaluation = RedirectModel(evaluation, prediction_model)
        print (evaluation)
        callbacks.append(evaluation)

        evaluation2 = CocoEval2(train_generator, tensorboard=tensorboard_callback, name_output=args.name_output)  # threshold(score)=0.05 defult
        evaluation2 = RedirectModel(evaluation2, prediction_model)
        print(evaluation2)
        callbacks.append(evaluation2)


    # save the model
    if args.save:
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(args.snapshot_path,'{}_{}_{{epoch:02d}}.h5'.format(args.backbone, args.dataset_type)),
            verbose=1,# save_best_only=True, monitor="mAP", mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))
    if args.tensorboard_dir:
        callbacks.append(tensorboard_callback)

    return callbacks

def create_generators(args, preprocess_image):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators..
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
    }

    if args.random_transform: # random transform generator for augmenting data
        transform_generator = random_transform_generator(
            min_rotation=0,
            max_rotation=math.pi/2,
            #min_rotation=-0.1,
            #max_rotation=0.1,
            #min_translation=(-0.1, -0.1),
            #max_translation=(0.1, 0.1),
           # min_shear=-0.1,
            #max_shear=0.1,
            #min_scaling=(0.9, 0.9),
           # max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            #contrast_range=(0.9, 1.1),
            #brightness_range=(-.1, .1),
            #hue_range=(-0.05, 0.05),
            #saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    trainfile=''
    testfile=''
    if args.dataset_type == 'coco':
        trainfile= 'train2017'
        testfile = 'val2017'
    elif args.dataset_type:
        trainfile = 'train'
        testfile = 'val'
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    train_generator = CocoGenerator(
        args.dataset_path,trainfile,
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        **common_args) # parameter to generator here
    validation_generator = CocoGenerator(
        args.dataset_path,testfile, shuffle_groups=False,
        **common_args)
    return train_generator, validation_generator

def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    """
    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))
    if parsed_args.multi_gpu > 1 and parsed_args.save:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))
    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk!")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))
    return parsed_args

def parse_args(args):
    parser     = argparse.ArgumentParser()
    parser.add_argument('--name_output', default='accloss', type=str)
    parser.add_argument('--dataset_type', default='g', type=str)
    parser.add_argument('--dataset_path', default='../../path/Detection_rota/demo_3images', type=str) #    'path/to/MS/COCO'
    #parser.add_argument('--resume',default ='../../test/trained_model/resnet50_trained_144.h5', type=str) #resnet50_256_0406.h5 ,default ='resnet50_256_0406', type=str
    #parser.add_argument('--resume', default=os.path.join('/home/io18230/Desktop/', 'resnet50_trained_144.h5'), type=str)
    parser.add_argument('--resume', default= None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save',              default=1, type=int, help='save model to snapshots')
    group.add_argument('--imagenet_weights',  help='Initialize the model with pretrained imagenet weights.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.', default = None)
    group.add_argument('--no_weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',         help='model retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch_size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi_gpu',        help='Number of GPUs to use parallel processing.', type=int, default=0)
    parser.add_argument('--multi_gpu_force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=3)#50
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=1) #10000
    parser.add_argument('--lr',               help=' ', type=float, default=1e-5)
    parser.add_argument('--nms_threshold',    help=' ', type=float, default=0.5)
    parser.add_argument('--snapshot_path',    help='Path to store snapshots of models during training', default='./snapshots')
    parser.add_argument('--tensorboard_dir',  help='Log directory for Tensorboard output') #default='./logs')
    parser.add_argument('--no_snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--evaluation',       help='per epoch evaluation.', default=1,type=int)
    #parser.add_argument('--freeze_backbone',  help='Freeze training of backbone layers.', default=1 ,action='store_false')
    parser.add_argument('--freeze_backbone',  help='Freeze training of backbone layers.', default=1,type=int)
    parser.add_argument('--random_transform', help='image and annotations.', default=1, type=int)
    parser.add_argument('--image_min_side',   help='Rescale the image so the smallest side is min_side.', type=int, default=256) #800
    parser.add_argument('--image_max_side',   help='.', type=int, default=342) # 1333
    parser.add_argument('--no_resize',        help='Don''t rescale the image.', action='store_true')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted_average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute_val_loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max_queue_size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)
    return check_args(parser.parse_args(args))

def save(his,args=None):
    if args is None: # parse arguments
        args = sys.argv[1:] # get args from outside --
    args = parse_args(args)

    new_dic = {}
    new_dic['info'] = {"description": "do not perform testing when loss is high", "year": 2021, "format": 0}
    new_dic['epoch'] = his.epoch
    new_dic['loss'] = his.history['loss']
    new_dic['val_loss'] = his.history['val_loss']

    new_dic['batch'] = []
    for id, item in enumerate(his.history['batch_loss']):
        new_dic['batch'].append([str(x) for x in item])

    if 'AP @[ IoU=0.50      | area=   all | maxDets=100 ]' in his.history.keys():
        new_dic['val_acc'] = his.history['AP @[ IoU=0.50      | area=   all | maxDets=100 ]']
        new_dic['val_acc7'] = his.history['AP @[ IoU=0.70      | area=   all | maxDets=100 ]']
    if 'AP @[ IoU=0.50      | area=   all | maxDets=10 ]' in his.history.keys():
        new_dic['acc'] = his.history['AP @[ IoU=0.50      | area=   all | maxDets=10 ]']
        new_dic['acc7'] = his.history['AP @[ IoU=0.70      | area=   all | maxDets=10 ]']

    json.dump(new_dic, open('{}.json'.format(args.name_output), 'w'), indent=4)
    print(his.history)


def main(args=None):
    global prediction_model
    check_keras_version()  # report error if not satisfied
    if args is None: # parse arguments
        args = sys.argv[1:] # get args from outside --
    args = parse_args(args)

    # 1 backbone
    backbone = models.backbone(args.backbone)# object stores backbone
    if args.gpu:     # optionally choose specific GPU
        setup_gpu(args.gpu)
    if args.config:  # optionally load config parameters  --didnot use it here
        print ('\n using defult cofigure')
        args.config = read_config_file(args.config)

    # 2 create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # 3 create the model
    if args.resume is not None:
        print('Resume training by loading model:'+args.resume+'\n')
        #print (kk) # when i am not busy, rewrite this part
        #kk = os.path.join(args.snapshot_path,'new','{}_{}_0{}.h5'.format(args.backbone, args.dataset_type, 2))
        #model = models.load_model(kk, backbone_name=args.backbone)
        model            = models.load_model(args.resume, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params,nms_threshold=args.nms_threshold)
        #prediction is a set of boxes, scores, lables and others
    else:
        weights = args.weights
        if weights is None and args.imagenet_weights:  # default to imagenet
            print ('\nInitialize the model with pretrained imagenet weights')
            weights = backbone.download_imagenet()
        else:
            print('\nInitialize the model with pretrained cow weights')

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,  # in models-init model with weights
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config,
            nms_threshold = args.nms_threshold
        )
    #print(model.summary())
    # lets the generator compute backbone layer shapes using the actual backbone model
    # if 'vgg' in args.backbone:
    #     train_generator.compute_shapes = make_shapes_callback(model)
    #     if validation_generator:
    #         validation_generator.compute_shapes = train_generator.compute_shapes

    # 4 create callback
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        train_generator, ###################################################
        args,
    )
    if not args.compute_val_loss:
        pass
        #validation_generator = None #using this now

    # 5 start training
    return training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data= validation_generator
    )

if __name__ == '__main__':
    his = main()
    save(his)

    #args = sys.argv[1:]   #get args from outside --  0: is the code it self
    #args = parse_args(args)

    #cd / home / io18230 / 0Projects / keras - retinanet - master / keras_retinanet
    #python bin/train.py --epochs=5 --steps=3 --resume=1 cow path/to/COW
    #python bin/train.py --epochs=2 --steps=3 cows path/to/COWs
    #python bin/train.py --epochs=50 --steps=4000 cows path/to/COWs