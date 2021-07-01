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

flags.DEFINE_string('framework', 'tf', '(tf)')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.7, 'score threshold')
flags.DEFINE_boolean('count', True, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # if FLAGS.output:
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(vid.get(cv2.CAP_PROP_FPS))
    #     # print("fpsssss: ", fps)
    #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    
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
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        # print("pred_bbox: ", pred_bbox)

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to allow detections for only people)
        #allowed_classes = ['person']

        # if crop flag is enabled, crop each detection and save it as new image
        #####################################################################################
        # if FLAGS.crop:
        #     crop_rate = 50#150 # capture images every so many frames (ex. crop photos every 150 frames)
        #     crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
        #     try:
        #         os.mkdir(crop_path)
        #     except FileExistsError:
        #         pass
        #     if frame_num % crop_rate == 0:
        #         final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
        #         try:
        #             os.mkdir(final_path)
        #         except FileExistsError:
        #             pass          
        #         crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
        #     else:
        #         pass
        ######################################################################################3

        # if FLAGS.count:
        #     # count objects found
        #     counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
        #     # loop through dict and print
        #     for key, value in counted_classes.items():
        #         print("Number of {}s: {}".format(key, value))
        #     image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        # else:
        #     image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate=FLAGS.plate)


        
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        # result = np.asarray(image)
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # if not FLAGS.dont_show:
            # cv2.imshow("result", result)
        
        # if FLAGS.output:
        #     out.write(result)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
        crop_rate = 50
        if frame_num % crop_rate == 0:
          result = utils.calculate(frame, pred_bbox, allowed_classes=allowed_classes)
          if result != None:
            result = np.asarray(result[0])
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            print("_______"+str(result[0]))
            print("________plate_number"+str(result[1]) )
            print("________confidence_score"+str(result[2]) )

    # cv2.destroyAllWindows()
    #---------------------------------------------------
    print("done!!!")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
