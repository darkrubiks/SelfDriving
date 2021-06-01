import os
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_LABELS = 'C:\\Users\\leona\\Desktop\\SelfDriving\\ObjectDetection\\kitti_label_map.pbtxt'
PATH_TO_CFG = 'C:\\Users\\leona\\Desktop\\SelfDriving\\ObjectDetection\\my_model\\pipeline.config'
PATH_TO_CKPT = 'C:\\Users\\leona\\Desktop\\SelfDriving\\ObjectDetection\\my_model\\checkpoint'


def load_model():
    """Load the detection model"""
    print('Loading model... ', end='')
    start_time = time.time()
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Model Loaded! Took {} seconds'.format(elapsed_time))

    return detection_model


@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


detection_model = load_model()  # load model

# load labels file
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
# image to detect
image_path = 'C:\\Users\\leona\\Desktop\\SelfDriving\\ObjectDetection\\images.jpg'
image_np = load_image_into_numpy_array(image_path)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

# detection
start_time = time.time()
detections = detect_fn(input_tensor, detection_model)
end_time = time.time()
elapsed_time = end_time - start_time
print('Detection Ready! Took {} seconds'.format(elapsed_time))

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.35
    ,
    agnostic_mode=False)

plt.figure(figsize=(15, 12))
plt.imshow(image_np_with_detections)
plt.axis('off')
plt.savefig('image_detected.png')
