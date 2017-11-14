"""
This file contains a function to upload local files to Amazon S3.
"""

import boto3
import os


def upload_file(filename_local, bucket_name, human_detected):
    """
    Upload local file to Amazon S3 and return its link.
    Depending on the image classification results, the file will be placed either
    in 'human' or 'false_positive' directory in the S3 bucket.
    The local copy will be deleted.

    :param filename_local: Local file
    :param bucket_name: Name of the S3 bucket
    :param human_detected: Result of the image classification
    :return: Link to the uploaded file
    """

    # Create an S3 client
    s3 = boto3.client('s3')

    # Filename that will appear in S3
    # Strip the image number at the beginning of the file name
    # as we want the file name to start with a date.
    # For example 04-20170724114420-00.jpg will become 20170724114420-00.jpg
    # The last two digits stand for the frame number.
    # http://htmlpreview.github.io/?https://github.com/Motion-Project/motion/blob/master/motion_guide.html#picture_filename
    # http://htmlpreview.github.io/?https://github.com/Motion-Project/motion/blob/master/motion_guide.html#conversion_specifiers
    filename_s3 = os.path.basename(filename_local)
    filename_s3 = filename_s3[filename_s3.find('-')+1:]

    if human_detected:
        filename_s3 = "human/{}".format(filename_s3)
    else:
        filename_s3 = "false_positive/{}".format(filename_s3)

    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    print("Uploading file {} to Amazon S3".format(filename_s3))
    s3.upload_file(filename_local, bucket_name, filename_s3, ExtraArgs={'ContentType': "image/jpeg", 'ACL': "public-read"})

    # Remove the image from the local file system
    print("Removing file {}".format(filename_local))
    os.remove(filename_local)

    # Generate url
    url = s3.generate_presigned_url('get_object', Params = {'Bucket': bucket_name, 'Key': filename_s3}, ExpiresIn = 7*24*3600)

    return url
