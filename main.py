import os
import random
from functools import partial
import torchvision.ops.boxes as bops

import cv2
import torch
import torchvision
import numpy as np
import multiprocessing
import time


# Good resources
# https://dsp.stackexchange.com/questions/37523/how-can-i-track-detected-objects-from-frame-to-frame
# https://stackoverflow.com/questions/15799487/parallel-image-detection-and-camera-preview-opencv-android
# https://stackoverflow.com/questions/53286749/run-two-process-simultaneously-in-python
# https://automaticaddison.com/how-to-do-multiple-object-tracking-using-opencv/

# Tracking
# https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-019-0646-0
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
# https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/#why-choose-the-opencv-dnn-module
# https://docs.opencv.org/4.5.2/d6/d0f/group__dnn.html

# Pretrained models:
# MobileNet v2:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt
# MobileNet v3:
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
# https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7

# Info:
# https://github.com/opencv/opencv/pull/16760
def process_images_rescale(train_images, train_annotations, size_scaled_img):
    """
    Rescale an image so that the shortest size is to size_scaled_img
    :param train_images: Images we want to rescale
    :param train_annotations: Annotations for the training images
    :param size_scaled_img: The max size of the shortest size of our images
    :return: list of scaled images and list of scaled annotations
    """
    processed_images = []
    scaled_annotations = []

    for img, annotations in zip(train_images, train_annotations):
        height, width, _ = img.shape
        scaled_annotations_image = []
        scale_h = size_scaled_img / height
        scale_w = size_scaled_img / width
        img = cv2.resize(img, (size_scaled_img, size_scaled_img))
        for annotation in annotations:
            scaled_annotation = np.array(
                [int(np.round(int(annotation[0]) * scale_w)), int(np.round(int(annotation[1]) * scale_h)),
                 int(np.round(int(annotation[2]) * scale_w)), int(np.round(int(annotation[3]) * scale_h))])
            scaled_annotations_image.append(scaled_annotation)

        processed_images.append(img)
        scaled_annotations.append(scaled_annotations_image)

    return processed_images, scaled_annotations


def divide_image(frame, split_width, split_height, pad_size, video_width, video_height):
    split_pairs_width = []
    split_pairs_height = []

    # Create tuples of indexes to split image with where there is overlap between the images
    for i in range(len(split_width) - 1):
        split_pairs_width.append((max(split_width[i] - pad_size, 0), min(split_width[i + 1] + pad_size, video_width)))
    for j in range(len(split_height) - 1):
        split_pairs_height.append(
            (max(split_height[j] - pad_size, 0), min(split_height[j + 1] + pad_size, video_height)))

    # Split image based on indexes
    # Added to list where widths over a given height are next to each other in the list, the stride size depends on the
    # Height split variable
    frame_parts = []
    for i, split_pair_width in enumerate(split_pairs_width):
        for j, split_pair_height in enumerate(split_pairs_height):
            subset = frame[split_pair_height[0]:split_pair_height[1], split_pair_width[0]:split_pair_width[1], :]
            frame_parts.append(subset)

    return frame_parts


def assemble_image(divided_image, split_width, split_height, pad_size):
    part_assembled = []
    num_splits_height = len(split_height)
    for i in range(num_splits_height, len(divided_image) + num_splits_height, num_splits_height - 1):
        temp_subset_1 = divided_image[i - num_splits_height]
        temp_subset_2 = divided_image[i - num_splits_height + 1]
        # Depending on where the chunks of image stand in the total image we remove parts where the images overlap
        # Height is always == 2, so this always trims the top image at the end of the index and the bottom image at the
        # Start of the index
        # For width the first two are trimmed at the end of the index, the middle are trimmed at the start and end and
        # The final two height joined pairs are only trimmed at the end if the image

        if i == num_splits_height:
            # These are at the left of
            divided_image_subset_1 = temp_subset_1[0:temp_subset_1.shape[0] - pad_size,
                                     0:temp_subset_1.shape[1] - pad_size]
            divided_image_subset_2 = temp_subset_2[pad_size:, 0:temp_subset_2.shape[1] - pad_size]

        if i == len(divided_image) + 1:
            divided_image_subset_1 = temp_subset_1[0:temp_subset_1.shape[0] - pad_size, pad_size:]
            divided_image_subset_2 = temp_subset_2[pad_size:, pad_size:]
        else:
            divided_image_subset_1 = temp_subset_1[0:temp_subset_1.shape[0] - pad_size,
                                     pad_size:temp_subset_2.shape[1] - pad_size]
            divided_image_subset_2 = temp_subset_2[pad_size:, pad_size:temp_subset_2.shape[1] - pad_size]

        height_wise_joined = np.concatenate((divided_image_subset_1, divided_image_subset_2),
                                            axis=0)
        part_assembled.append(height_wise_joined)
    return np.concatenate(part_assembled, axis=1)


def load_fine_tuned_model(model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    if device.type == 'cuda':
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    model = torch.load(model_path, map_location=device)

    if device.type == 'cuda':
        model.cuda()
    model.eval()
    return model


def main_video():
    num_cores = multiprocessing.cpu_count()
    # print(len(os.sched_getaffinity(0)))
    width_divide = 1
    height_divide = 1

    if num_cores > 4:
        width_divide = 2
        height_divide = 2
    # elif num_cores == 8 or num_cores > 8:
    #     width_divide = 4
    #     height_divide = 2

    tensor_transform = torchvision.transforms.ToTensor()
    model_path = 'fine_tuned_model_pytorch/chk_point10'
    model = load_fine_tuned_model(model_path)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,
    #                                                                            pretrained_backbone=True,
    #                                                                            trainable_backbone_layers=0)
    model.eval()

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, test_frame = capture_frame(video_capture)
    video_height, video_width = test_frame.shape[:2]

    # Appending -1 to the indexes to split with to allow for easy iteration
    split_width = list(range(0, video_width, video_width // width_divide))
    split_width.append(video_width)
    split_height = list(range(0, video_height, video_height // height_divide))
    split_height.append(video_height)
    pad_size = 128

    pool = multiprocessing.Pool(processes=4)
    forward_pass = partial(perform_forward_pass, model=model, tensor_transform=tensor_transform)
    processes = []
    running = True

    while True:
        t0 = time.time()

        ret, frame = capture_frame(video_capture)
        divided_frame = divide_image(frame, split_width, split_height, pad_size, video_width, video_height)
        # perform_forward_pass(divided_frame[0], model, tensor_transform)
        boxes = []
        # if processes:
        #     for p in processes:
        #         if not p.is_alive():
        #             running=False
        boxes = pool.map_async(forward_pass, divided_frame).get()
        # if not running:
        #     for i in range(len(divided_frame)):
        #         p = pool.apply_async(forward_pass, (divided_frame[i],))
        #         processes.append(p)
        #     for process in processes:
        #         boxes.append(process.get())
        print(boxes)

        # for sub_frame in divided_frame:
        #     perform_forward_pass(sub_frame, model, tensor_transform)
        t1 = time.time()
        print(t1 - t0)
        assembled_frame = assemble_image(divided_frame, split_width, split_height, pad_size)

        frame = assembled_frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            cv2.destroyAllWindows()
            break

        t1 = time.time()
    video_capture.release()
    cv2.destroyAllWindows()


def capture_frame(vid_capture):
    # Capture frame-by-frame
    ret, frame = vid_capture.read()
    return ret, frame


def within_bounding_box(point, box):
    # Check if outside of box instead of within to slightly speed up the check
    not_within_box = point[0] < box[0] or point[1] < box[1] or point[0] > box[2] or point[1] > box[3]
    return not not_within_box


def convert_bbox_to_cv2_format(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    box_cv2_format = (box[0], box[1], width, height)
    return box_cv2_format


def get_box_per_part(divided_frames, model, transform, split_height, split_width, pad_size, start_end_list,
                     start, end):
    for i, frame in enumerate(divided_frames):
        boxes = perform_forward_pass(frame, model, transform)
        if boxes:
            # To check for overlap take two bounding boxes, calculate overlap and see if overlap is equal to width *
            # padsize!
            for box in boxes:
                if i == 0:
                    start = [int(box[0]), int(box[1])]
                    end = [int(box[2]), int(box[3])]
                if i == 1:
                    start = [int(box[0]), int(box[1]) + split_height[1] - pad_size]
                    end = [int(box[2]), int(box[3]) + split_height[1] - pad_size]
                if i == 2:
                    start = [int(box[0]) + split_width[1] - pad_size, int(box[1])]
                    end = [int(box[2]) + split_width[1] - pad_size, int(box[3])]
                if i == 3:
                    start = [int(box[0]) + split_width[1] - pad_size,
                             int(box[1]) + split_height[1] - pad_size]
                    end = [int(box[2]) + split_width[1] - pad_size, int(box[3]) + split_height[1] - pad_size]

                start.extend(end)
                start_end_list.append(start)
    return start_end_list


def update_tracked_bboxes(start_end_list, objects_to_track):
    current_ids = objects_to_track.keys()
    for bounding_box in start_end_list:
        if within_bounding_box(posList, bounding_box):
            if bounding_box in objects_to_track.values():
                objects_to_track = {key: val for key, val in objects_to_track.items() if val != bounding_box}

            else:
                while True:
                    random_id = random.randint(0, 100000)
                    if random_id not in current_ids:
                        break
                objects_to_track[random_id] = bounding_box
    return objects_to_track


def perform_forward_pass_concurrent(q_in, q_out, model, tensor_transform):
    while True:
        print('running')
        frame = q_in.get()
        prediction_dict = model([tensor_transform(frame)])
        predicted_boxes = []
        if prediction_dict[0]['scores'].shape[0] > 0:
            print('Highest score: {}'.format(prediction_dict[0]['scores'][0]))
            # print(prediction_dict)
            for object_num, score in enumerate(prediction_dict[0]['scores']):
                if score > .34:
                    predicted_boxes.append(prediction_dict[0]['boxes'][object_num].detach().numpy())

        q_out.put(predicted_boxes)
        print('Finished')

def perform_forward_pass(frame, model, tensor_transform):
    prediction_dict = model([tensor_transform(frame)])
    predicted_boxes = []
    num_objects = 0
    if prediction_dict[0]['scores'].shape[0] > 0:
        # print('Highest score: {}'.format(prediction_dict[0]['scores'][0]))
        for object_num, score in enumerate(prediction_dict[0]['scores']):
            if score > .5:
                predicted_boxes.append(prediction_dict[0]['boxes'][object_num].detach().numpy())
                num_objects += 1
    return predicted_boxes

def perform_forward_pass_opencv(frame, model):


def main_object_detection(q_in, q_out, model, transform, split_width, split_height,
                          pad_size, video_width, video_height):
    q_split_in = multiprocessing.Queue()
    q_split_out = multiprocessing.Queue(4)
    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=perform_forward_pass_concurrent,
                                    args=(q_split_in, q_split_out, model, transform))
        p.start()
        processes.append(p)

    while True:
        start = (0, 0)
        end = (0, 0)

        boxes_split = []
        frame = q_in.get()
        if frame == 'exit':
            print('Shutting down object detection')
            break

        divided_frames = divide_image(frame, split_width, split_height,
                                      pad_size, video_width, video_height)
        if q_split_in.empty():
            for sub_frame in divided_frames:
                q_split_in.put(sub_frame)

        if q_split_out.qsize() == 4:
            for i in range(4):
                boxes_split.append(q_split_out.get())
            if boxes_split:
                start = (int(boxes_split[0][0][0]), int(boxes_split[0][0][1]))
                end = (int(boxes_split[0][0][2]), int(boxes_split[0][0][3]))

            q_out.put((start, end))


def main_object_detection_serial(q_in, q_out, model, transform, split_width, split_height,
                                 pad_size, video_width, video_height):
    while True:
        start = [(0, 0)]
        end = [(0, 0)]

        start_end_list = []
        frame = q_in.get()

        if frame == 'exit':
            print('Shutting down object detection')
            break

        # Move this to other process
        divided_frames = divide_image(frame, split_width, split_height,
                                      pad_size, video_width, video_height)
        # Gets boxes of split input image
        start_end_list = get_box_per_part(divided_frames, model, transform, split_height, split_width,
                                          pad_size, start_end_list, start, end)

        # Merge boxes that overlap a lot, still need to implement recursive ish merging and fine-tune
        # The min amount of overlap
        indices_to_remove = []
        for i in range(len(start_end_list)):
            for j in range(i, len(start_end_list)):
                if i != j:
                    start_i = start_end_list[i][0:2]
                    start_j = start_end_list[j][0:2]
                    end_i = start_end_list[i][2:]
                    end_j = start_end_list[j][2:]
                    inter, union = bops._box_inter_union(torch.as_tensor([start_end_list[i]]),
                                                         torch.as_tensor([start_end_list[j]]))
                    min_overlap = pad_size * max(end_i[0] - start_i[0],
                                                 end_j[0] - start_j[0])
                    if inter > min_overlap:
                        new_box = [min(start_i[0], start_j[0]), min(start_i[1], start_j[1]),
                                   max(end_i[0], end_j[0]), max(end_i[1], end_j[1])]

                        start_end_list.append(new_box)
                        indices_to_remove.append(i)
                        indices_to_remove.append(j)
        # Not completely right, cant merge two merged bounding boxes, but is edge case so..
        for index in sorted(set(indices_to_remove), reverse=True):
            del (start_end_list[index])

        q_out.put(start_end_list)


def main_video_co_occurring(model, transform):
    do_object_tracking = False
    do_object_detection = True
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def mouse_press(event, x, y, flags, param):
        global posList
        if event == cv2.EVENT_LBUTTONDOWN:
            posList = [x, y]

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_press)

    start_list = [(0, 0)]
    end_list = [(0, 0)]

    # Check if we can divide the image into multiple parts
    width_divide = 1
    height_divide = 1
    if multiprocessing.cpu_count() > 4:
        width_divide = 2
        height_divide = 2

    # Set up video capture and find height and width of video
    ret, frame = capture_frame(video_capture)
    video_height, video_width = frame.shape[:2]

    # Appending -1 to the indexes to split with to allow for easy iteration
    # Here we find the image splits based on number of parts we split in
    split_width = list(range(0, video_width, video_width // width_divide))
    split_width.append(video_width)
    split_height = list(range(0, video_height, video_height // height_divide))
    split_height.append(video_height)

    # Padding is to allow the algorithm to detect objects that are 'split' by the tiling object detection
    pad_size = 64

    # p_process = multiprocessing.Process(target=main_object_detection,
    #                                     args=(q_in, q_out, model, transform, split_width,
    #                                           split_height, pad_size,
    #                                           video_width, video_height))
    p_process = multiprocessing.Process(target=main_object_detection_serial,
                                        args=(q_in, q_out, model, transform, split_width,
                                              split_height, pad_size,
                                              video_width, video_height))
    p_process.start()
    q_in.put(frame)

    start_end_list = []
    trackers = []
    objects_to_track = {}
    multi_tracker = cv2.legacy.MultiTracker_create()

    while True:
        t0 = time.time()

        ret, frame = capture_frame(video_capture)
        # Code that detects if a key is pressed
        key = cv2.waitKey(1)

        # Code to quit camera and stop object detection
        if key & 0xFF == ord('q'):
            print("Turning off camera.")
            video_capture.release()
            print("Camera off.")
            cv2.destroyAllWindows()
            q_in.put('exit')
            break

        # If we still perform object detection we first check if the input queue for object detection is empty
        # Then if q_out has element we retrieve the detected boxes and update the start, end values to 0. This will
        # Remove found bounding boxes if no objects are detected. We then iterate over found bounding boxes
        # And draw them
        if do_object_detection:
            if q_in.empty():
                q_in.put(frame)
            if not q_out.empty():
                start_list = [(0, 0)]
                end_list = [(0, 0)]
                start_end_list = q_out.get()
            if start_end_list:
                for i in range(len(start_end_list)):
                    if start_end_list[i]:
                        start = start_end_list[i][:2]
                        end = start_end_list[i][2:]
                        start_list.append(start)
                        end_list.append(end)
                        # Bug: When you press pause one frame disappears due to start and end being only one value
                        # Fix: Make it a list?
                        frame = cv2.rectangle(frame, start, end, (0, 255, 0), 2)

        if do_object_tracking:
            # Update the location of the bounding boxes
            success, bboxes = multi_tracker.update(frame)

            # Draw the bounding boxes on the video frame
            for i, bbox in enumerate(bboxes):
                start = (int(bbox[0]), int(bbox[1]))
                end = (int(bbox[0] + bbox[2]),
                           int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, start, end, (0, 255, 0), 2)
        # If pressed 'p' object detection will pause and the last bounding boxes are displayed
        # The video capture does keep running
        if key & 0xFF == ord('p'):
            if do_object_detection:
                do_object_detection = False
            else:
                do_object_detection = True

        # Code for handling mouse pressed and adding an objects bounding boxes in to track array
        if not do_object_detection and posList:
            if start_end_list:
                objects_to_track = update_tracked_bboxes(start_end_list, objects_to_track)
            posList.clear()

        # Here we start the object tracking, first check if there are objects to track
        if key & 0xFF == ord('s'):
            if objects_to_track:
                do_object_tracking = True
                do_object_detection = False
                for num_object, (id_object, object_box) in enumerate(objects_to_track.items()):
                    # tracker = cv2.legacy.TrackerKCF_create()
                    tracker = cv2.legacy.TrackerCSRT_create()
                    cv2_box = convert_bbox_to_cv2_format(object_box)
                    multi_tracker.add(tracker, frame, cv2_box)
                    start_list = []
                    end_list = []
                print('Starting Object tracking')
            else:
                print('No objects to track')

        for start, end in zip(start_list, end_list):
            frame = cv2.rectangle(frame, start, end, (0, 255, 0), 2)

        cv2.imshow('image', frame)

    video_capture.release()
    cv2.destroyAllWindows()
    p_process.join()


if __name__ == '__main__':
    tensor_transform = torchvision.transforms.ToTensor()
    model_path = 'fine_tuned_model_pytorch/chk_point10'
    model_loaded = load_fine_tuned_model(model_path)
    posList = []

    main_video_co_occurring(model_loaded, tensor_transform)
