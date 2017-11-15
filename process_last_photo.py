#!/usr/bin/env python

import ConfigParser
import os
from utils.tf_serving_client import *
from utils.upload_file import *
from utils.send_email import *


def main():
    # Read config file
    config = ConfigParser.ConfigParser()
    config.read('process_last_photo.cfg')

    img_dir = config.get('Motion', 'img_dir', "/home/pi/motion")
    server = config.get('Serving', 'server', "localhost:9000")
    bucket_name = config.get('S3', 'bucket', "rpizero-smart-camera-archive")
    user = config.get('Email', 'user')
    pwd = config.get('Email', 'pwd')

    # Find the latest image
    files = os.listdir(img_dir)
    full_paths = [os.path.join(img_dir, basename) for basename in files]
    filename_local = max(full_paths, key=os.path.getctime)

    # Identify objects in the picture using TensorFlow Serving
    classes, scores = query_mobilenet_server(filename_local, server)
    human_detected = False
    print("\n".join(["{0}: {1:.2f}".format(c, s) for (c, s) in zip(classes, scores)]))
    for (c, s) in zip(classes, scores):
        if c == 'person' and s > 0.5:
            human_detected = True
            break
    print("human_detected = {}".format(human_detected))

    # Upload file to S3 and remove the local copy
    url = upload_file(filename_local, bucket_name, human_detected)

    # Send e-mail notification
    if human_detected:
        subject = "Human detected"
        body = "\n".join(["{0}: {1:.2f}".format(c, s) for (c, s) in zip(classes, scores)])
        body += "\n\n{}".format(url)
        send_email(user, pwd, user, subject, body)


if __name__== "__main__":
     main()
