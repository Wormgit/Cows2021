import keras, json
from ..utils.coco_eval import evaluate_coco

dic={}
dic['info'] = {"description": "log every epoch", "year": 2020, "format": 0}
dic['epoch'] = []
dic['loss'] = []
dic['val_loss'] = []
dic['batch'] = []
dic['val_acc'] = []
dic['val_acc7'] = []
dic['acc'] = []
dic['acc7'] = []

class CocoEval(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """
    def __init__(self, generator, tensorboard=None, threshold=0.05, name_output = '0'):
        """ CocoEval callback intializer.
        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard
        self.name_output = name_output
        self.tmp = []

        super(CocoEval, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    #'AP @[ IoU=0.55      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.60      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.65      | area=   all | maxDets=100 ]',
                     'AP @[ IoU=0.70      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.80      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.85      | area=   all | maxDets=100 ]',
                    # 'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    # 'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    # 'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                  #  'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                  #  'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                  #  'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                   # 'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    #'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                   # 'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
                    ]
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold, self.name_output)

        logs['batch_loss'] = self.tmp
        self.tmp = []
        if coco_eval_stats is not None:
            for index, result in enumerate(coco_eval_stats):
                logs[coco_tag[index]] = result

            if self.tensorboard:
                import tensorflow as tf
                if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                    summary = tf.compat.v1.Summary()
                    for index, result in enumerate(coco_eval_stats):
                        summary_value = summary.value.add()
                        summary_value.simple_value = result
                        summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                        self.tensorboard.writer.add_summary(summary, epoch)

            dic['epoch'].append(epoch)
            dic['loss'].append(logs['loss'])
            dic['val_loss'].append(logs['val_loss'])

            t = []
            for item in logs['batch_loss']:
                t.append(str(item))
            dic['batch'].append(t)

            if 'AP @[ IoU=0.50      | area=   all | maxDets=100 ]' in logs.keys():
                dic['val_acc'].append(logs['AP @[ IoU=0.50      | area=   all | maxDets=100 ]'])
                dic['val_acc7'].append(logs['AP @[ IoU=0.70      | area=   all | maxDets=100 ]'])
            if 'AP @[ IoU=0.50      | area=   all | maxDets=10 ]' in logs.keys():
                dic['acc'].append(logs['AP @[ IoU=0.50      | area=   all | maxDets=10 ]'])
                dic['acc7'].append(logs['AP @[ IoU=0.70      | area=   all | maxDets=10 ]'])
            json.dump(dic, open('e_{}.json'.format(self.name_output), 'w'), indent=4)

            # this is for training loss!!!!!!
    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        self.tmp.append(logs.get('loss'))

class CocoEval2(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """
    def __init__(self, generator, tensorboard=None, threshold=0.05, name_output = '0'):
        """ CocoEval callback intializer.
        Args
            generator   : The generator used for creating validation data.
            tensorboard : If given, the results will be written to tensorboard.
            threshold   : The score threshold to use.
        """
        self.generator = generator
        self.threshold = threshold
        self.tensorboard = tensorboard
        self.name_output = name_output

        super(CocoEval2, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=10 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=10 ]',
                    'AP @[ IoU=0.70      | area=   all | maxDets=10 ]',
                    ]
        coco_eval_stats = evaluate_coco(self.generator, self.model, self.threshold, self.name_output)

        if coco_eval_stats is not None:
            for index, result in enumerate(coco_eval_stats):
                logs[coco_tag[index]] = result

            if self.tensorboard:
                import tensorflow as tf
                if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                    summary = tf.compat.v1.Summary()
                    for index, result in enumerate(coco_eval_stats):
                        summary_value = summary.value.add()
                        summary_value.simple_value = result
                        summary_value.tag = '{}. {}'.format(index + 1, coco_tag[index])
                        self.tensorboard.writer.add_summary(summary, epoch)
