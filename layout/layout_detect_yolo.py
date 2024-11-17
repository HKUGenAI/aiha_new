# https://github.com/opendatalab/PDF-Extract-Kit/blob/main/pdf_extract_kit/tasks/layout_detection/models/yolo.py

import os
import cv2
import torch
import yaml
import numpy as np
from tqdm import tqdm
from utils.visualization import visualize_bbox

class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title', 
            1: 'plain text',
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption', 
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'isolate_formula', 
            9: 'formula_caption'
        }

        # Load the YOLO model from the specified path
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(config['model_path'])
        except AttributeError:
            from ultralytics import YOLO
            self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def predict(self, images, result_path, image_ids=None):
        """
        Predict formulas in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        results = []
        for idx, image in enumerate(tqdm(images, desc="Predicting layout")):
            result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False, device=self.device)[0]
            if self.visualize:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                boxes = result.__dict__['boxes'].xyxy
                classes = result.__dict__['boxes'].cls
                scores = result.__dict__['boxes'].conf

                if self.iou_thres > 0:
                    indices = self.nms_func(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=self.iou_thres)
                    boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
                    if len(boxes.shape) == 1:
                        boxes = np.expand_dims(boxes, 0)
                        scores = np.expand_dims(scores, 0)
                        classes = np.expand_dims(classes, 0)
                
                vis_result = visualize_bbox(image, boxes, classes, scores, self.id_to_names)

                # Determine the base name of the image
                if image_ids:
                    base_name = image_ids[idx]
                else:
                    # base_name = os.path.basename(image)
                    base_name = os.path.splitext(os.path.basename(image))[0]  # Remove file extension
                
                result_name = f"{base_name}_layout.png"
                
                # Save the visualized result                
                cv2.imwrite(os.path.join(result_path, result_name), vis_result)
            results.append(result)
        return results
    
if __name__ == "__main__":
    with open('configs/layout_yolo.yaml') as f:
        config = yaml.safe_load(f)
    layout_detector = LayoutDetectionYOLO(config["model_config"])
    images = ['inputs/EngineeringatHKUACenturyofExcellence_page_34.png']
    result_path = 'results'

    results = layout_detector.predict(images, result_path)
    print(results[0].__dict__)

    from layout_sort import sort_with_layout

    for result in results:
        bboxes = result.boxes
        page_shape = result.orig_shape

        text_bboxes = [b for i, b in enumerate(bboxes.data) if bboxes.cls[i] in [0, 1]]
        
        # Sort the bounding boxes based on their layout
        sorted_bboxes, layout_bbox = sort_with_layout(text_bboxes, page_shape[1], page_shape[0])
        
        print(sorted_bboxes)
        print(layout_bbox)
    
    


