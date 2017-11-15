"""
This file contains a functions for TensorFlow Serving client requests.
"""

from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import os
from PIL import Image
import numpy as np

from object_detection.utils import label_map_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/tmp/vendored', 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

label_map = label_map_util.load_labelmap("mscoco_label_map.pbtxt")
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
    """
    Helper function to return image as a numpy array.
    """

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def query_incpetion_server(filename, server):
    """
    Query the server running the TensorFlow Serving of the Inception-v3 model.

    :param filename: JPEG image filename
    :param server: Server address in the format host:port
    :return: Array of classes and scores from the image classification
    """

    host, port = server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    with open(filename, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=[1]))
        result = stub.Predict(request, 10.0)  # 10 secs timeout

        classes = result.outputs['classes'].string_val
        scores = result.outputs['scores'].float_val
        return classes, scores


def query_mobilenet_server(filename, server):
    """
    Query the server running the TensorFlow Serving of the SDD MobileNet v1 model.

    :param filename: JPEG image filename
    :param server: Server address in the format host:port
    :return: Array of classes and scores from the image classification
    """

    host, port = server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    with Image.open(filename) as image:
        # See prediction_service.proto for gRPC request/response details.
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'ssd_mobilenet_v1_coco'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image_np_expanded, 
                shape=image_np_expanded.shape, dtype='uint8'))
        result = stub.Predict(request, 10.0)  # 10 secs timeout

        # The classification gives 100 objects. Return only top 10.
        classes_id = result.outputs['detection_classes'].float_val[:10]
        classes = [category_index[c]['name'] for c in classes_id]
        scores = result.outputs['detection_scores'].float_val[:10]
        return classes, scores
