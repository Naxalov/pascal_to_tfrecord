import io
import json
import os
import pathlib

from absl import app
from absl import flags
from absl import logging
import hashlib
from lxml import etree
import PIL.Image
import tensorflow as tf
import untangle
import tfrecord_util

from tfrecord_util import *
class UniqueId:
  """Class to get the unique {image/ann}_id each time calling the functions."""

  def __init__(self):
    self.image_id = 0
    self.ann_id = 0

  def get_image_id(self):
    self.image_id += 1
    return self.image_id

  def get_ann_id(self):
    self.ann_id += 1
    return self.ann_id



def xml_to_tf_example(data_dir,unique_id,xml_obj):

    full_path = xml_obj.annotation.path.cdata
    filename = xml_obj.annotation.filename.cdata
    print(full_path)
    with tf.io.gfile.GFile(f'{data_dir}/JPEGImages/{full_path}', 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    print(image.format)

    image_id = unique_id.get_image_id()

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(xml_obj.annotation.size.width.cdata)
    height = int(xml_obj.annotation.size.height.cdata)

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    area = []
    classes_text = []
    classes = []
    truncated = []
    poses = []
    difficult_obj = []

    for obj in xml_obj.annotation.object:

        xmin.append(float(obj.bndbox.xmin.cdata) / width)
        ymin.append(float(obj.bndbox.ymin.cdata) / height)
        xmax.append(float(obj.bndbox.xmax.cdata) / width)
        ymax.append(float(obj.bndbox.ymax.cdata) / height)
        area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
        classes_text.append(obj.name.cdata.encode('utf8'))
        classes.append(0)
        print(obj.truncated.cdata.encode('utf8'))
        truncated.append(int(obj.truncated.cdata.encode('utf8')))
        difficult_obj.append(int(obj.difficult.cdata.encode('utf8')))
        poses.append(obj.pose.cdata.encode('utf8'))
        


    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename.encode('utf8')),
        'image/source_id': bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        
        'image/object/area':float_list_feature(area),
        'image/object/class/text':bytes_list_feature(classes_text),
        'image/object/class/label':int64_list_feature(classes),
        'image/object/difficult':int64_list_feature(difficult_obj),
        'image/object/truncated':int64_list_feature(truncated),
        'image/object/view':bytes_list_feature(poses),
    }))
    return example


data_dir = 'Raccoon'

tfrecord_path = 'train.tfrecord'

writer = tf.io.TFRecordWriter(tfrecord_path)

unique_id = UniqueId()
annotations_dir = os.path.join(data_dir, 'Annotations')
examples_list = os.listdir(annotations_dir)
for idx, example in enumerate(examples_list):
    print(example)
    if example.endswith('.xml'):
        path = os.path.join(annotations_dir,example)
        xml_obj = untangle.parse(path)
        # print(xml_obj)
        tf_example = xml_to_tf_example(data_dir,unique_id,xml_obj)
        # print(tf_example)
        writer.write(tf_example.SerializeToString())

writer.close()