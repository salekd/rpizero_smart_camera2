"""
This file contains a function for TensorFlow Serving client requests.
"""

from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def query_tf_server(filename, server):
    """
    Query the server running the TensorFlow Serving of the Inception-v3 model.

    :param filename: JPEG image filename
    :param server: Server address in the format host:port
    :return: PredictResponse object with the image classification results. 
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

        return result
