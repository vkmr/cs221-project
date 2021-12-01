import os
from typing_extensions import ParamSpec
import copy

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

MIN_IOU_FOR_ENSAMBLE = 0.7
MIN_CONFIDENCE_FOR_SINGLE = 0.2

class Detection:
    def __init__(self, class_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence):
        self.class_id = int(class_id)
        self.top_left_x = float(top_left_x)
        self.top_left_y = float(top_left_y)
        self.bottom_right_x = float(bottom_right_x)
        self.bottom_right_y = float(bottom_right_y)
        self.confidence = float(confidence)
    def __repr__(self) -> str:
        return f"{self.class_id} {round(self.confidence, 4)} {self.top_left_x} {self.top_left_y} {self.bottom_right_x} {self.bottom_right_y}"


def ensamble_two(model_1_detections: 'list[Detection]', model_2_detections: 'list[Detection]', min_iou_for_ensamble: 'float'):
    ensambled_detections = []
    matched_model_1_detection_indices = []
    matched_model_2_detection_indices = []
    for i1, d1 in enumerate(model_1_detections):
        max_iou = None        
        bb1 = {'x1': d1.top_left_x, 'y1': d1.top_left_y, 'x2': d1.bottom_right_x, 'y2': d1.bottom_right_y }
        for i2, d2 in enumerate(model_2_detections):
            bb2 = {'x1': d2.top_left_x, 'y1': d2.top_left_y, 'x2': d2.bottom_right_x, 'y2': d2.bottom_right_y }
            iou = get_iou(bb1, bb2)            
            if (max_iou is None or iou > max_iou) and d1.class_id == d2.class_id:
                max_iou = iou
                ensambled_detection = copy.copy(d1) if d1.confidence > d2.confidence else copy.copy(d2)
                ensambled_detection.confidence = (d1.confidence + d2.confidence) / 2.0 # TODO: Is iou factor a good idea?
                matched_model_1_detection_index = i1
                matched_model_2_detection_index = i2
        if max_iou is not None and max_iou >= min_iou_for_ensamble:
            ensambled_detections.append(ensambled_detection)
            matched_model_1_detection_indices.append(matched_model_1_detection_index)
            matched_model_2_detection_indices.append(matched_model_2_detection_index)
    
    for i1, d1 in enumerate(model_1_detections):
        if i1 not in matched_model_1_detection_indices:
            if d1.confidence > MIN_CONFIDENCE_FOR_SINGLE:
                ensambled_detection = copy.copy(d1)
                ensambled_detection.confidence /= 2.0
                ensambled_detections.append(ensambled_detection)

    for i2, d2 in enumerate(model_2_detections):
        if i2 not in matched_model_2_detection_indices:
            if d2.confidence > MIN_CONFIDENCE_FOR_SINGLE:
                ensambled_detection = copy.copy(d2)
                ensambled_detection.confidence /= 2.0
                ensambled_detections.append(ensambled_detection)
    
    return ensambled_detections




def parse_detections_per_image(image_file) -> 'list[Detection]':
    detections = []
    with open(image_file) as file:
        for line in file:
            if(len(line.split()) == 6):
                detection = Detection(*line.split())
                detections.append(detection)
    return detections

def parse_detections_all_images(images_root_dir) -> 'list[list[Detection]]':
    detections = []
    for image_file in os.listdir(images_root_dir):
        image_detections = parse_detections_per_image(os.path.join(images_root_dir,image_file))
        detections.append(image_detections)
    return detections

def get_image_names_without_extension(images_root_dir):
    image_names = []
    for image_file in os.listdir(images_root_dir):
        image_names.append(os.path.splitext(image_file)[0])
    return image_names

def main():
    image_names_without_extension = get_image_names_without_extension('C:/Users/Mahbub/Desktop/cs221-project/v5l6/labels')
    model_1_detections = parse_detections_all_images('C:/Users/Mahbub/Desktop/cs221-project/v5l6/labels')
    model_2_detections = parse_detections_all_images('C:/Users/Mahbub/Desktop/cs221-project/v5s/labels')
    ensambled_detection_root = 'C:/Users/Mahbub/Desktop/cs221-project/mAP-master/input/detection-results'

    ensambled_all_images_detections = []
    for image_name_without_extension, image_detections_1, image_detections_2 in zip(image_names_without_extension, model_1_detections, model_2_detections):
        ensambled_image_detections = ensamble_two(image_detections_1, image_detections_2, MIN_IOU_FOR_ENSAMBLE)
        ensambled_all_images_detections.append(ensambled_image_detections)
        ensambled_detections_file_path = os.path.join(ensambled_detection_root, image_name_without_extension + ".txt")
        with open(ensambled_detections_file_path, "w") as ensambled_detections_file:
            for an_ensambled_detection in ensambled_image_detections:
                ensambled_detections_file.write(an_ensambled_detection.__repr__())
                ensambled_detections_file.write("\n")

    return

if __name__ == "__main__":
    main()



