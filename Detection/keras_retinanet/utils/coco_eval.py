from ..bin.cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import keras
import numpy as np
import json

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def evaluate_coco(generator, model, threshold=0.05, name=0):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in progressbar.progressbar(range(generator.size()), prefix='Rotate evaluation: '):
        image = generator.load_image(index)
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        # correct boxes for image scale
        boxes[:,:,:-1] /= scale ####################################################
        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # compute predicted labels and scores
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                break
            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_coco_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
            }

            # append detection to results
            results.append(image_result)

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name+name), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name+name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name+name))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats
