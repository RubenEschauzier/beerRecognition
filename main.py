import cv2
import os
import tensorflow as tf
import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import config_util


def load_fine_tuned_model(pipeline_config, num_classes):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _prediction_heads=detection_model._box_predictor._prediction_heads,
        _box_predictor_ = detection_model._box_predictor._box_prediction_head
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        # _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)

    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore('fine_tuned_model/ckpt-21')
    return detection_model

def load_mobilenet_ssd(num_classes, check_point, pipeline_config):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)
    # print(tf.train.list_variables(tf.train.latest_checkpoint(check_point)))
    # print(detection_model._box_predictor._prediction_heads)

    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _prediction_heads=detection_model._box_predictor._prediction_heads['box_encodings'],
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        # _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(os.path.join(check_point, 'ckpt-0'))

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 300, 300, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    return detection_model


def main_video():
    # model = load_fine_tuned_model('ssd_mobilenet/pipeline.config', 2)
    pipeline_config = 'ssd_mobilenet/pipeline.config'
    checkpoint_path = 'ssd_mobilenet/checkpoint'
    model = load_mobilenet_ssd(2, checkpoint_path, pipeline_config)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture_frame(video_capture)
        image, shapes = model.preprocess(tf.expand_dims(tf.convert_to_tensor(frame, dtype=tf.float32), 0))
        prediction_dict = model.predict(image, shapes)
        test = model.postprocess(prediction_dict, shapes)
        start = (int(test['detection_boxes'][0][0][0]*300), int(test['detection_boxes'][0][0][1]*300))
        end = (int(test['detection_boxes'][0][0][2]*300), int(test['detection_boxes'][0][0][3]*300))
        thickness = 2
        color = (0, 255, 0)
        frame = cv2.rectangle(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), start, end, color, thickness)
        print(test.keys())
        print(np.max(test['detection_scores'].numpy()))

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            cv2.destroyAllWindows()
            break

    video_capture.release()
    cv2.destroyAllWindows()


def capture_frame(vid_capture):
    # Capture frame-by-frame
    ret, frame = vid_capture.read()
    return ret, frame


if __name__ == '__main__':
    main_video()
