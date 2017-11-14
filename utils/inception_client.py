"""
This file contains the functions for TensorFlow Serving client requests.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def inception_client_main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    with open(FLAGS.image, 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.model_spec.signature_name = 'predict_images'
        request.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=[1]))
        result = stub.Predict(request, 10.0)  # 10 secs timeout
        print(result)

        return result


def query_tf_server(filename_local, server):
    tf.app.flags.DEFINE_string('image', filename_local)
    tf.app.flags.DEFINE_string('server', server)
    FLAGS = tf.app.flags.FLAGS
    tf.app.run(inception_client_main)
