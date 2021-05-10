# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:23:09 2021

@author: Christoph
"""

# first tell python where all of the directories are for tf - NB do this in anaconda prompt window
import sys
#sys.path += [r"C:\Users\Christoph\Documents\Python Scripts\chequeclearance\scripts\RealTimeObjectDetection-main\Tensorflow\models\research",
#             r"C:\Users\Christoph\Documents\Python #Scripts\chequeclearance\scripts\RealTimeObjectDetection-main\Tensorflow\models\research\object_detection",
             #r"C:\Users\Christoph\Documents\Python Scripts\chequeclearance\scripts\RealTimeObjectDetection-main\Tensorflow\models",
             #r"C:\Users\Christoph\Documents\Python Scripts\chequeclearance\scripts\RealTimeObjectDetection-main\Tensorflow\models\official"]


# load in required object detection libraries
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


# paths for saved model
os.chdir(r'C:\Users\Christoph\Documents\Python Scripts\chequeclearance\tensorflow_flask')
PATH_TO_SAVED_MODEL = 'my_ssd_effdet_d1/'

# load model inference graph and label directory
detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL+'saved_model')
# Get model labels
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_SAVED_MODEL+'label_map.pbtxt')


# run inference for a single image and show - option 1
def run_inference_for_single_image(model, image, score_threshold):
  image = np.array(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]
  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy() 
  return output_dict

def show_inference(model, image_path, score_threshold):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np, score_threshold)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      min_score_thresh=score_threshold,
      line_thickness=4)
  #for displaying in notebook
  #display(Image.fromarray(image_np))
  # for displaying with cv2 (note resize)
  height, width = image_np.shape[:2]
  print(height,width)
  cv2.namedWindow('im_test', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('im_test', width, height)
  cv2.imshow('im_test', image_np)
  cv2.waitKey(0)
  cv2.imwrite(IMAGE_PATH+'_labelled.jpg', image_np)
  #Image.show(Image.fromarray(image_np))
  #plt.imshow(image_np)


IMAGE_PATH = r'C:\Users\Christoph\Documents\Python Scripts\chequeclearance\scripts\RealTimeObjectDetection-main\Tensorflow\workspace\images\out_of_sample'
image_path = IMAGE_PATH+'/random_sigs.JPG'
show_inference(detection_model, image_path, 0.2)