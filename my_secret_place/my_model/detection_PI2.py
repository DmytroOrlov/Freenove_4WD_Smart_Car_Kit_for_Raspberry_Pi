#!/usr/bin/env python
# coding: utf-8

# In[107]:


import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)


# In[2]:


#!python -m pip install --upgrade pip


# ## Load the model

# In[108]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
#import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
#import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
#import tensorflow as tf
import time



# In[38]:


from picamera.array import PiRGBArray
from picamera import PiCamera
import time


# In[109]:


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


# In[110]:


tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)


# In[111]:


DATA_DIR = os.path.join(os.getcwd(), 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)


# ## Load the model

# In[ ]:





# In[122]:


MODELS_DIR = "/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/my_secret_place/my_model"
PATH_TO_CFG = os.path.join(MODELS_DIR,  'pipeline.config')
PATH_TO_CKPT = os.path.join(MODELS_DIR,  'checkpoint/')
PATH_TO_LABELS = "/Freenove_4WD_Smart_Car_Kit_for_Raspberry_Pi/my_secret_place/my_model/label_map.pbtxt"


# In[123]:


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


# In[114]:


configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)


# tf.train.Checkpoint - Manages saving/restoring trackable values to disk.
# TensorFlow objects may contain trackable state, such as tf.Variables, tf.keras.optimizers.Optimizer implementations, tf.data.Dataset iterators, tf.keras.Layer implementations, or tf.keras.Model implementations. These are called trackable objects.
# A Checkpoint object can be constructed to save either a single or group of trackable objects to a checkpoint file. It maintains a save_counter for numbering checkpoints.
# Checkpoint.save() and Checkpoint.restore() write and read object-based checkpoints

# In[115]:


# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


# In[116]:


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# In[117]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


# In[95]:


video_path = "/home/jovyan/Toys/MyBarbie.mp4"


# In[118]:


cap = cv2.VideoCapture(0)  # for webcam (0)


# In[119]:


flag = "pi"


# In[120]:


def detection_func(image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
        #################################################
    tf.config.run_functions_eagerly(True)     #Enables / disables eager execution of tf.functions.
      #################################################    
    
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    
        ##################################################
    tf.config.run_functions_eagerly(False)
        ########################################
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    scores = detections['detection_scores'][0].numpy()
    output = list(zip(classes, scores))
    result = list(filter(lambda x: x[1] > 0.1, output))
    
   # print(result)
    
    
#    #display  #please, hide it for pi
#     viz_utils.visualize_boxes_and_labels_on_image_array(
#             image_np_with_detections,
#             detections['detection_boxes'][0].numpy(),
#             (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
#             detections['detection_scores'][0].numpy(),
#             category_index,
#             use_normalized_coordinates=True,
#             max_boxes_to_draw=200,
#             min_score_thresh=.30,
#             agnostic_mode=False)
#    # time.sleep(2)
#     cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

#     if cv2.waitKey(25) & 0xFF == ord('q'):
          

#         cap.release()
#         cv2.destroyAllWindows()
    
   


# In[121]:


if flag == "pc":
    while True:
        # Read frame from camera
   # for i in range(1):
        ret, image_np = cap.read()
        
        detection_func(image_np)
        

        
elif flag == "pi":
 
    camera = PiCamera()
    rawCapture = PiRGBArray(camera)
    time.sleep(0.1)
    camera.capture(rawCapture, format="bgr")
    #while True:
    for i in range(10):
        image = rawCapture.array
        detection_func(image)
        time.sleep(2)
    


# In[ ]:




