import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from core.local_utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import glob
import pytesseract
import re
import datetime


def load_model(path):
  try:
    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects={})
    model.load_weights('%s.h5' % path)
    return model
  except Exception as e:
    print(e)

def preprocess_image(img, resize=False):
  img = img / 255
  if resize:
      img = cv2.resize(img, (224,224))
  return img

def get_plate(image_path, Dmax=608, Dmin = 608):
  wpod_net_path = "data/model/wpod-net.json"
  wpod_net = load_model(wpod_net_path)
  vehicle = preprocess_image(image_path)
  ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
  side = int(ratio * Dmin)
  bound_dim = min(side, Dmax)
  _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
  return vehicle, LpImg, cor

def ocr(img):
  plate_image = cv2.convertScaleAbs(img[0], alpha=(255.0))
  gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(7,7),0)

  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  blur = cv2.medianBlur(thresh, 3)
  blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
  # cv2.imwrite("blur.jpg", blur)
  try:
    text = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    text = re.sub('[\W_]+', '', str(text))
  except: 
    text = None
  return text

def calculate(image, bboxes):
  classes = read_class_names(cfg.YOLO.CLASSES)
  num_classes = len(classes)
  image_h, image_w, _ = image.shape
  out_boxes, out_scores, out_classes, num_boxes = bboxes
  if num_boxes > 0:
    for i in range(num_boxes):
      xmin, ymin, xmax, ymax = out_boxes[i]

      score = out_scores[i]
      cropped_img = image[int(ymin)-50:int(ymax)+50, int(xmin)-60:int(xmax)+60]
      vehicle, plate, cor = get_plate(cropped_img)
      if len(cor) > 0:
        plate_number = ocr(plate)
        num_len = len(plate_number)
        print("text lengh: ", num_len)
        print("text: ", plate_number)
        plate_number = re.sub('[\W_]+', '', str(plate_number))
        
        if ( num_len == 5 or num_len == 6 ):
          cv2.imwrite("static/images/cropped_{}.jpg".format(i, str), cropped_img)
          with open("static/texts/plate_number_{}.txt".format(i, str), "w") as text_file:
            print(f"{str(plate_number)}", file=text_file)
          # d1 = datetime.now()
          # with open("static/texts/time_{}.txt".format(i, str), "w") as text_file:
          #   print(f"{str(d1)}", file=text_file)

def main(video_input):
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
  input_size = 416 
  video_path = video_input 

  video_name = video_path.split('/')[-1]
  video_name = video_name.split('.')[0]

  saved_model_loaded = tf.saved_model.load('checkpoints/custom-416', tags=[tag_constants.SERVING])
  infer = saved_model_loaded.signatures['serving_default']

  try:
      vid = cv2.VideoCapture(int(video_path))
  except:
      vid = cv2.VideoCapture(video_path)

  frame_num = 0
  plate_number = None
  confidence_score = None
  while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        image = Image.fromarray(frame)
    else:
        print('Video has ended or failed, try a different video format!')
        break

    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
      boxes = value[:, :, 0:4]
      pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.7
    )

    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    print("frame: ", frame_num)
    
    calculate(frame, pred_bbox)
  print("done!!!")
  return "ok"

# if __name__ == '__main__':
#   try:
#       app.run(main("./data/video/test1.mp4"))
#   except SystemExit:
#       pass
