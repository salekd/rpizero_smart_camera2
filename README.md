# rpizero_smart_camera2

TensorFlow Serving

https://www.tensorflow.org/serving/setup

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install tensorflow-model-server-universal
```

TensorFlow

https://petewarden.com/2017/08/20/cross-compiling-tensorflow-for-the-raspberry-pi/

```
sudo apt-get install libblas-dev liblapack-dev python-dev libatlas-base-dev gfortran python-setuptools
sudo pip2 install http://ci.tensorflow.org/view/Nightly/job/nightly-pi-zero/lastSuccessfulBuild/artifact/bazel-out/pi/tensorflow-1.3.0-cp27-none-any.whl
```

```
sudo apt-get install libjpeg-dev
sudo pip install Pillow
```

```
git clone https://github.com/tensorflow/models.git
```

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled.

```
sudo apt-get install protobuf-compiler
cd models
protoc object_detection/protos/*.proto --python_out=.
```

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -zxvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
```
