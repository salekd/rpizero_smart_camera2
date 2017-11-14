#!/usr/bin/env python

import ConfigParser
import utils


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
    res = query_tf_server(filename_local, server)


    # Upload file to S3 and remove the local copy
    url = upload_file(filename_local, bucket_name, human_detected)

    # Send e-mail notification
    if human_detected:
        send_email(user, pwd, user, subject, body)


if __name__== "__main__":
     main()
