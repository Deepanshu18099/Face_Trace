"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from skimage import io, transform
import facenet
import align.detect_face
import random
from time import sleep

# Hyperparameters and constants
MINSIZE = 20  # minimum size of face
THRESHOLD = [0.7, 0.7, 0.75]  # three steps' threshold
FACTOR = 0.709  # scale factor
DEFAULT_IMAGE_SIZE = 182
DEFAULT_MARGIN = 44
DEFAULT_GPU_MEMORY_FRACTION = 1.0
DEFAULT_DETECT_MULTIPLE_FACES = True

def parse_arguments(argv):
    """
    Parse the command-line arguments and return them as a namespace.
    
    Args:
    argv (list): List of command-line arguments.
    
    Returns:
    argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE, 
                        help='Image size (height, width) in pixels.')
    parser.add_argument('--margin', type=int, default=DEFAULT_MARGIN, 
                        help='Margin for the crop around the bounding box (height, width) in pixels.')
    parser.add_argument('--random_order', action='store_true',
                        help='Shuffles the order of images to enable alignment using multiple processes.')
    parser.add_argument('--gpu_memory_fraction', type=float, default=DEFAULT_GPU_MEMORY_FRACTION,
                        help='Upper bound on the amount of GPU memory that will be used by the process.')
    parser.add_argument('--detect_multiple_faces', type=bool, default=DEFAULT_DETECT_MULTIPLE_FACES,
                        help='Detect and align multiple faces per image.')
    return parser.parse_args(argv)

def initialize_mtcnn(gpu_memory_fraction):
    """
    Initialize the MTCNN face detector with TensorFlow session.

    Args:
    gpu_memory_fraction (float): The fraction of GPU memory allocated for the process.

    Returns:
    tuple: Initialized pnet, rnet, and onet for face detection.
    """
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            return align.detect_face.create_mtcnn(sess, None)

def align_and_save_face(image_path, output_filename, pnet, rnet, onet, image_size, margin, detect_multiple_faces):
    """
    Detect and align faces from an image and save the aligned face thumbnails.

    Args:
    image_path (str): Path to the input image.
    output_filename (str): Path to save the aligned face image.
    pnet, rnet, onet: MTCNN networks for face detection.
    image_size (int): The size of the output aligned image.
    margin (int): Margin around the detected face bounding box.
    detect_multiple_faces (bool): Whether to detect and align multiple faces in the image.

    Returns:
    bool: True if a face was successfully aligned and saved, False otherwise.
    """
    try:
        img = io.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        print(f'Error reading {image_path}: {e}')
        return False

    if img.ndim < 2:
        return False
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:, :, 0:3]

    bounding_boxes, _ = align.detect_face.detect_face(img, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    nrof_faces = bounding_boxes.shape[0]
    img_size = np.asarray(img.shape)[0:2]

    if nrof_faces > 1:
        print(f'Found {nrof_faces} faces in {image_path}')
    elif nrof_faces == 0:
        print(f'No faces found in {image_path}')
        return False
    if nrof_faces > 0:
        det_arr = []
        det = bounding_boxes[:, 0:4]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # Weight on centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        count = 0
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            print(np.min(cropped),np.max(cropped))
            scaled = transform.resize_local_mean(cropped, (image_size, image_size))
            scaled = (scaled)/ (scaled.max())
            print(np.min(scaled),np.max(scaled))
            try:
                io.imsave(output_filename + f'_{i}.png', scaled)
                count += 1
            except (IOError, ValueError, IndexError) as e:
                print(f'Error saving {output_filename}_{i}: {e}')
        return count

def process_dataset(dataset, output_dir, pnet, rnet, onet, image_size, margin, detect_multiple_faces, random_order):
    """
    Process an entire dataset of images, detecting and aligning faces from each image.

    Args:
    dataset (list): List of dataset objects with image paths.
    output_dir (str): Directory to save the aligned images.
    pnet, rnet, onet: MTCNN networks for face detection.

    image_size (int): The size of the output aligned images.
    margin (int): Margin around the detected face bounding box.
    detect_multiple_faces (bool): Whether to detect and align multiple faces in the images.
    
    random_order (bool): Whether to shuffle the images before processing.

    Returns:
    tuple: Number of successfully aligned images and total number of images.
    """
    nrof_images_total = 0
    nrof_successfully_aligned = 0

    if random_order:
        random.shuffle(dataset)

    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        if random_order:
            random.shuffle(cls.image_paths)

        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename)

            if not os.path.exists(output_filename + '_0.png'):
                success = align_and_save_face(image_path, output_filename, pnet, rnet, onet, image_size, margin, detect_multiple_faces)
                if success:
                    nrof_successfully_aligned += success

    return nrof_successfully_aligned, nrof_images_total

def main(args):
    """
    Main function to perform face detection and alignment on a dataset.
    
    Args:
    args (argparse.Namespace): Parsed command-line arguments.
    """
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    facenet.store_revision_info(os.path.dirname(__file__), output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)

    pnet, rnet, onet = initialize_mtcnn(args.gpu_memory_fraction)

    nrof_successfully_aligned, nrof_images_total = process_dataset(dataset, output_dir, pnet, rnet, onet, args.image_size, args.margin, args.detect_multiple_faces, args.random_order)

    print(f'Total number of images: {nrof_images_total}')
    print(f'Number of successfully aligned images: {nrof_successfully_aligned}')
    return nrof_successfully_aligned

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    # something to end
    
