import math
import time
import datetime
from random import randint

import pandas as pd
import numpy as np
import os
import tarfile

import torchvision as torchvision
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import xml.etree.ElementTree as ET
from natsort import natsorted

import torch
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#from PyTorchHelpers.vision.references.detection.engine import evaluate


def extract_all_tar():
    """

    :return: Nothing, just extracts tar file directories, this takes a while
    """
    with tarfile.open('image_data.tar') as tar:
        tar.extractall(members=tar.getmembers())


def load_train_data(glass_folder, bottle_folder, annotations_glasses, annotations_bottles):
    """
    Function to load training data and convert to RGB format if needed. It checks if there are annotations available
    for the images

    :param glass_folder: Folder from extracted tar files that contains images of beer glasses
    :param bottle_folder: Folder from extract tar files that contains images of beer bottles
    :param annotations_glasses: Files of beer glasses that have annotations attached to them
    :param annotations_bottles: Files of beer bottles that have annotations attached to them
    :return: A list of numpy representations of images of beer glasses and bottles
    """
    files_glasses = natsorted(os.listdir(glass_folder))
    files_bottles = natsorted(os.listdir(bottle_folder))

    # Check if our images our annotated, we remove the .JPEG to compare names of images
    files_glasses = [file for file in files_glasses if file.split('.')[0] in annotations_glasses]
    files_bottles = [file for file in files_bottles if file.split('.')[0] in annotations_bottles]

    list_images_glass = []
    list_images_bottles = []

    for glass_file in files_glasses:
        image = Image.open(glass_folder + r'/' + glass_file)
        image_array = np.array(image)
        # Some images are grayscale, convert to RGB
        if len(image_array.shape) <= 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        list_images_glass.append(image_array)

    for bottle_file in files_bottles:
        image = Image.open(bottle_folder + r'/' + bottle_file)
        image_array = np.array(image)

        # Some images are grayscale, convert to RGB
        if len(image_array.shape) <= 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        list_images_bottles.append(image_array)

    return list_images_glass, list_images_bottles


def load_ground_truth(annotation_folder_glass, annotation_folder_bottle):
    # Useful: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
    # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    # https://heartbeat.fritz.ai/real-time-object-detection-using-ssd-mobilenet-v2-on-video-streams-3bfc1577399c
    # https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
    # https://stackoverflow.com/questions/48915003/get-the-bounding-box-coordinates-in-the-tensorflow-object-detection-api-tutorial
    # https://towardsdatascience.com/centernet-explained-a7386f368962
    files_glasses = natsorted(os.listdir(annotation_folder_glass))
    files_bottles = natsorted(os.listdir(annotation_folder_bottle))

    list_annotation_glasses = []
    list_annotation_bottles = []

    for glass_file, bottle_file in zip(files_glasses, files_bottles):
        glass_tree_root = ET.parse(annotation_folder_glass + r'/' + glass_file).getroot()
        bottle_tree_root = ET.parse(annotation_folder_bottle + r'/' + bottle_file).getroot()

        bndbox_glass = []
        bndbox_bottle = []
        for object_found in glass_tree_root.iter('object'):
            box = []
            for coord in object_found.find('bndbox'):
                box.append(int(coord.text))

            bndbox_glass.append(np.array(box))

        for object_found in bottle_tree_root.iter('object'):
            box = []
            for coord in object_found.find('bndbox'):
                box.append(int(coord.text))

            bndbox_bottle.append(np.array(box))

        list_annotation_glasses.append(bndbox_glass)
        list_annotation_bottles.append(bndbox_bottle)

    files_bottles = [file.split('.')[0] for file in files_bottles]
    files_glasses = [file.split('.')[0] for file in files_glasses]

    return list_annotation_glasses, list_annotation_bottles, files_glasses, files_bottles


def load_validation_annotations(val_csv_path):
    val_solutions = pd.read_csv(val_csv_path)
    # Dictionary will contain: File name to open for image, list with [class, bounding boxes]
    validation_image_box_pairs = {}

    for i in range(val_solutions.shape[0]):
        # Iterate over dataframe of validation solutions and extract bounding boxes, split string into parts where each
        # Bounding box entry is 1 category name and 4 coordinates
        bounding_boxes = val_solutions['PredictionString'][i].split()
        glass_boxes = []
        bottle_boxes = []

        glass_class = []
        bottle_class = []

        for j in range(0, len(bounding_boxes), 5):
            # Find relevant validation images
            if bounding_boxes[j] == 'n02823750':
                string_bbox = bounding_boxes[j + 1:j + 5]
                glass_boxes.append([int(coord) for coord in string_bbox])
                glass_class.append([1])
            if bounding_boxes[j] == 'n02823428':
                string_bbox = bounding_boxes[j + 1:j + 5]
                bottle_boxes.append([int(coord) for coord in string_bbox])
                bottle_class.append([2])
        # If we found any images of glasses we add to the the dictionary with one hot encoded classes
        if len(glass_boxes) > 0:
            validation_image_box_pairs[val_solutions['ImageId'][i]] = [glass_class, glass_boxes]

        if len(bottle_boxes) > 0:
            validation_image_box_pairs[val_solutions['ImageId'][i]] = [bottle_class, bottle_boxes]

    return validation_image_box_pairs


def load_validation_data(validation_images_folder, validation_images_dict):
    files_validation = os.listdir(validation_images_folder)
    files_to_extract = [x + '.JPEG' for x in validation_images_dict.keys()]

    validation_images_list = []
    validation_annotations_list = []
    validation_classes_list = []
    for file in files_validation:
        if file in files_to_extract:
            image = Image.open(validation_images_folder + r'/' + file)
            image_array = np.array(image)

            # Some images are grayscale, convert to RGB
            if len(image_array.shape) <= 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            assert len(image_array.shape) == 3

            key_to_open = file.split('.')[0]

            validation_annotation = np.array(validation_images_dict[key_to_open][1])
            validation_class = validation_images_dict[key_to_open][0]

            validation_images_list.append(image_array)
            validation_annotations_list.append(validation_annotation)
            validation_classes_list.append(validation_class)

    return validation_images_list, validation_annotations_list, validation_classes_list


def pre_process_data(images_glasses, images_bottle, annotations_glasses, annotations_bottles, min_size):
    """
    :param images_glasses: A list of np arrays that represent training images of beer glasses
    :param images_bottle: A list of np array that represent training images of beer bottles
    :param min_size: Minimum number of pixels the small side should have
    :return: A list with all images that have a shortest side above min_size pixels
    """

    # Create ground truths of the classes by one-hot encoding
    ground_truth_glasses = [[1 for y in range(len(annotations_glasses[x]))] for x in
                            range(len(annotations_glasses))]
    ground_truth_bottles = [[2 for y in range(len(annotations_bottles[x]))] for x in
                            range(len(annotations_bottles))]

    # Add all annotations and images together
    images_glass.extend(images_bottles)
    annotations_glasses.extend(annotations_bottles)
    ground_truth_glasses.extend(ground_truth_bottles)

    # For clarity
    train_images_list = images_glass
    train_annotations_list = annotations_glasses
    train_classes = ground_truth_glasses
    return train_images_list, train_annotations_list, train_classes


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
        if height == width:
            scale = size_scaled_img / height
            img = cv2.resize(img, (size_scaled_img, size_scaled_img), fx=scale, fy=scale)
            for annotation in annotations:
                scaled_annotation = np.array([int(np.round(int(coord) * scale)) for coord in annotation])
                scaled_annotations_image.append(scaled_annotation)

        if height > width:
            scale = size_scaled_img / width
            new_width = math.floor(scale * height)
            img = cv2.resize(img, (size_scaled_img, new_width), fx=scale, fy=scale)

            for annotation in annotations:
                scaled_annotation = np.array([int(np.round(int(coord) * scale)) for coord in annotation])
                scaled_annotations_image.append(scaled_annotation)

        if width > height:
            scale = size_scaled_img / height
            new_height = math.floor(scale * width)
            img = cv2.resize(img, (new_height, size_scaled_img), fx=scale, fy=scale)
            for annotation in annotations:
                scaled_annotation = np.array([int(np.round(int(coord) * scale)) for coord in annotation])
                scaled_annotations_image.append(scaled_annotation)

        processed_images.append(img)
        scaled_annotations.append(scaled_annotations_image)

    return processed_images, scaled_annotations


def process_images_sample_randomly(train_images, train_annotations, size_input_img):
    """
    Function to randomly sample a size_input_img x size_input_img square from our training data, this ensures all
    data is equally as big

    :param train_images: The input processed training images
    :param train_annotations: The input annotations that were previously scaled
    :param size_input_img: The desired size of our output training samples
    :return: The rescaled training samples
    """
    processed_images = []
    processed_annotations = []
    for img, annotations in zip(train_images, train_annotations):
        # Create indices for maximum index of starting point of subsample so that the left over range is still equal to
        # size_input_img

        index_w, index_h = img.shape[0] - size_input_img, img.shape[1] - size_input_img
        for annotation in annotations:
            if index_w >= annotation[3]:
                index_w = index_w // 2
            if index_h >= annotation[1]:
                index_h = index_h // 2
        # print(img.shape[0])
        # print(img.shape[1])
        # print(index_w)
        # print(index_h)
        # Create range randomly to sample from image from
        start_index_w, start_index_h = randint(0, index_w), randint(0, index_h)
        for annotation in annotations:
            if start_index_w + size_input_img <= annotation[1]:
                start_index_w += (annotation[3] - annotation[1])
            if start_index_h + size_input_img <= annotation[0]:
                start_index_h += (annotation[2] - annotation[0])


        processed_annotations_image = []
        for annotation in annotations:
            new_x0 = min(max(0, annotation[0] - start_index_h), size_input_img)
            new_y0 = min(max(0, annotation[1] - start_index_w), size_input_img)
            new_x1 = min(annotation[2] - start_index_h, size_input_img)
            new_y1 = min(annotation[3] - start_index_w, size_input_img)

            # new_box = [annotation[0]-start_index_h, annotation[1]-start_index_w, annotation[2]-start_index_h,
            #           annotation[3]-start_index_w]
            new_box = np.array([new_x0, new_y0, new_x1, new_y1])
            processed_annotations_image.append(new_box)

        # Reshape image and check if it has correct dimensions
        reshape_img = img[start_index_w:start_index_w + size_input_img, start_index_h:start_index_h + size_input_img]
        assert reshape_img.shape == (size_input_img, size_input_img, 3)

        processed_images.append(reshape_img)
        processed_annotations.append(processed_annotations_image)

    return processed_images, processed_annotations


def plot_image_and_bounding_box(training_image, bounding_box):
    img = training_image
    thickness = 2
    color = (0, 255, 0)

    for box in bounding_box:
        start = (int(box[0]), int(box[1]))
        end = (int(box[2]), int(box[3]))
        img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), start, end, color, thickness)

    cv2.imshow('Rectangle example', img)
    cv2.waitKey(0)


def define_parameters():
    glass_id = 1
    bottle_id = 2
    num_classes = 2

    category_index = {glass_id: {'id': glass_id, 'name': 'Beer Glass'},
                      bottle_id: {'id': bottle_id, 'name': 'Beer Bottle'}}

def inference_on_fine_tuned_model(model_loc, val_images):
    pass


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # Format as hh:mm:ss


# From pytorch tutorial utils.py from https://github.com/pytorch/vision/tree/master/references/detection
def collate_fn(batch):
    return tuple((zip(*batch)))


def train_pytorch_model(dataset_train, dataset_val, num_classes, num_epoch):
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,
                                                                               trainable_backbone_layers=6)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    if device.type == 'cuda':
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    if device.type == 'cuda':
        model.cuda()

    training_stats = []
    total_t0 = time.time()

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epoch):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epoch))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0

        model.train()
        transform_test = torchvision.transforms.ToPILImage()

        for step, batch in enumerate(data_loader_train):
            if step % 30 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader_train), elapsed))

            images = [image.to(device) for image in batch[0]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            # for img, dict in zip(images,targets):
            #     plot_image_and_bounding_box(np.array(transform_test(img)), np.array(transform_test(dict['boxes'])))
            #     print(dict['labels'])

            model.zero_grad()
            # print([target['boxes']for target in targets])
            # print(images[0].shape)

            forward_output = model(images, targets)
            total_loss = sum(loss for loss in forward_output.values())
            total_train_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            lr_scheduler.step()

        average_train_loss = total_train_loss / len(data_loader_train)
        train_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.4f}".format(average_train_loss))
        print("  Training epoch took: {:}".format(train_time))

        ckpoint_name = 'chk_point{}'.format(epoch)
        directory = os.path.join('fine_tuned_model_pytorch', ckpoint_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(model, directory + '/model_fine_tuned')
        print('Saved model checkpoint to \'{}\''.format(directory))

        # model.eval()
        # total_eval_loss = 0
        #
        # for step, batch in enumerate(data_loader_val):
        #     if step % 30 == 0 and not step == 0:
        #         elapsed = format_time(time.time() - t0)
        #         print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader_train), elapsed))
        #
        #     images = [image.to(device) for image in batch[0]]
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
        #
        #     forward_output = model(images)
        #     print(forward_output)
        #
        #     for img, dict in zip(images, forward_output):
        #         plot_image_and_bounding_box(np.array(transform_test(img)), [(dict['boxes']).detach().numpy()[0]])
        #         print(dict['labels'][0])
        #
        #     break


class BeerDataset(torch.utils.data.Dataset):
    def __init__(self, img, box_list, classes, sample_function=None, size_sample=None):
        self.images = img
        self.box_list = box_list
        self.classes = classes
        self.sample_function = sample_function
        self.size_sample = size_sample
        self.tensor_transform = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = self.images[idx]
        bounding_box = self.box_list[idx]
        labels = self.classes[idx]

        if isinstance(idx, int):
            images = [images]
            bounding_box = [bounding_box]
            labels = [labels]
        #
        # if self.sample_function and self.size_sample:
        #     #print('Before sample: {}'.format(bounding_box))
        #     batch_images, [batch_boxes] = self.sample_function(images, bounding_box, self.size_sample)
        #     #print('After sample: {}'.format(batch_boxes))
        # else:
        batch_images = images
        [batch_boxes] = bounding_box

        batch_labels = torch.squeeze(torch.as_tensor(labels), 0)

        # BUG? Does this need transform like this?
        #batch_boxes = torch.squeeze(self.tensor_transform(np.array(batch_boxes)), 0)
        batch_boxes = torch.as_tensor(np.array(batch_boxes))

        # Still need to fix these labels!!
        batch_labels = torch.reshape(torch.tensor(labels), (batch_boxes.shape[0],))
        #batch_labels = torch.ones((batch_boxes.shape[0],), dtype=torch.int64)

        batch_area = (batch_boxes[:, 3] - batch_boxes[:, 1]) * (batch_boxes[:, 2] - batch_boxes[:, 0])

        if len(batch_images) == 1:
            [batch_images] = batch_images
            batch_images = self.tensor_transform(batch_images)
        sample = {'boxes': batch_boxes, 'labels': batch_labels, 'area': batch_area, 'image_id': torch.tensor([idx]),
                  'iscrowd': torch.zeros((batch_boxes.shape[0],), dtype=torch.int64)}
        return batch_images, sample

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # use pytorch
    # https://towardsdatascience.com/building-your-own-object-detector-pytorch-vs-tensorflow-and-how-to-even-get-started-1d314691d4ae
    folder_glass = r'ILSVRC/Data/CLS-LOC/train/n02823750'
    folder_bottle = r'ILSVRC/Data/CLS-LOC/train/n02823428'
    annotations_glass = r'ILSVRC/Annotations/CLS-LOC/train/n02823750'
    annotations_bottle = r'ILSVRC/Annotations/CLS-LOC/train/n02823428'

    validation_csv = r'imagenet-data/LOC_val_solution.csv'
    validation_folder = r'ILSVRC/Data/CLS-LOC/val'

    size_first_scale_step = 224
    size_training_input = 300

    validation_file_dict = load_validation_annotations(validation_csv)
    validation_images, validation_annotations, validation_classes = load_validation_data(validation_folder,
                                                                                         validation_file_dict)
    scaled_val_images, scaled_val_annotations = process_images_rescale(validation_images, validation_annotations,
                                                                       size_training_input)

    annotation_glasses, annotation_bottles, annotated_glasses, annotated_bottles = \
        load_ground_truth(annotations_glass, annotations_bottle)
    images_glass, images_bottles = load_train_data(folder_glass, folder_bottle, annotated_glasses, annotated_bottles)

    training_images, training_annotations, training_classes = \
        pre_process_data(images_glass, images_bottles, annotation_glasses, annotation_bottles, size_first_scale_step)

    scaled_training_images, scaled_training_annotations = process_images_rescale(training_images, training_annotations,
                                                                                 size_first_scale_step)
    training_images = training_images[:len(training_annotations)]

    train_dataset = BeerDataset(training_images, training_annotations, training_classes,
                                process_images_sample_randomly, 224)

    validation_dataset = BeerDataset(scaled_val_images, scaled_val_annotations, validation_classes,
                                     process_images_sample_randomly, 224)


    train_pytorch_model(train_dataset, validation_dataset, 3, 100)
    # processed_train_images, processed_train_annotations = process_images_sample_randomly(scaled_training_images,
    #                                                                                      scaled_training_annotations,
    #                                                                                      size_training_input)

    # https://stackoverflow.com/questions/51220865/resize-bounding-box-according-to-image
# Good thing on preprocessing https://stackoverflow.com/questions/40744700/how-can-i-find-imagenet-data-labels
