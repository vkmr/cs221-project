import json
import os

class Detection:
    def __init__(self, class_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        self.class_id = int(class_id)
        self.top_left_x = float(top_left_x)
        self.top_left_y = float(top_left_y)
        self.bottom_right_x = float(bottom_right_x)
        self.bottom_right_y = float(bottom_right_y)
    def __repr__(self) -> str:
        return f"{self.class_id} {self.top_left_x} {self.top_left_y} {self.bottom_right_x} {self.bottom_right_y}"

def annotations_to_detections(annotations_json_file_path, detection_root):
    annotations_json_file = open(annotations_json_file_path)
    annotations = json.load(annotations_json_file)

    image_id_to_name = {}
    images = annotations['images']
    for image in images:
        image_id_to_name[image["id"]] = image["file_name"] 

    ground_truth_detections_per_image = {}
    for annotation in annotations['annotations']:
        image_name = os.path.splitext(image_id_to_name[annotation['image_id']])[0]
        
        if image_name not in ground_truth_detections_per_image:
            ground_truth_detections_per_image[image_name] = []
        ground_truth_detections_per_image[image_name].append(Detection(
            annotation['category_id'] - 1, # Off by 1 error correction
            annotation['bbox'][0],
            annotation['bbox'][1],
            annotation['bbox'][0] + annotation['bbox'][2],
            annotation['bbox'][1] + annotation['bbox'][3],
        ))
    
    for image_name, ground_truth_detections_for_image in ground_truth_detections_per_image.items():
        ground_truth_detection_file_path = os.path.join(detection_root, image_name + ".txt")
        with open(ground_truth_detection_file_path, "w") as ground_truth_detection_file:
            for a_ground_truth_detection in ground_truth_detections_for_image:
                ground_truth_detection_file.write(a_ground_truth_detection.__repr__())
                ground_truth_detection_file.write("\n")



def main():
    annotations_json_file_path = "C:/Users/Mahbub/Desktop/cs221-project/annotations.coco.json"
    ground_truth_detection_root = 'C:/Users/Mahbub/Desktop/cs221-project/mAP-master/input/ground-truth'
    annotations_to_detections(annotations_json_file_path, ground_truth_detection_root)

if __name__ == "__main__":
    main()
