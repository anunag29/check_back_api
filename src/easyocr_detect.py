import easyocr
from PIL import Image
import cv2
import numpy as np

class EasyOCRDetect:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])  

    def extract_bounding_boxes(self, image: Image.Image) -> list[dict]:
        image_np = np.array(image)
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        
        results = self.reader.readtext(image_np)
        # print("readtext ended")
        bounding_boxes = []
        for (bbox, _, _) in results:
            bounding_boxes.append({
                'bbox': bbox  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # 'text': text,
                # 'prob': prob
            })
        
        image_height = image.size[1]  
        threshold = image_height * 0.05 
        
        # Group bounding boxes by y-coordinate proximity
        grouped_boxes = self.group_boxes_by_line(bounding_boxes, threshold)
        return grouped_boxes

    def group_boxes_by_line(self, bounding_boxes: list[dict], threshold: float) -> list[dict]:
        # Sort bounding boxes by their y1 coordinate
        bounding_boxes.sort(key=lambda x: min(y for x, y in x['bbox']))
        
        merged_boxes = []
        current_group = []
        
        for box in bounding_boxes:
            x1, y1 = box['bbox'][0]
            x2, y2 = box['bbox'][2]
            
            if not current_group:
                current_group.append(box)
            else:
                _, last_y1 = current_group[-1]['bbox'][0]
                _, last_y2 = current_group[-1]['bbox'][2]
                
                # If the current box is close enough to the previous one in terms of y-coordinates
                if abs(y1 - last_y1) < threshold and abs(y2 - last_y2) < threshold:
                    current_group.append(box)
                else:
                    merged_boxes.append(self.merge_group(current_group))
                    current_group = [box]
        
        if current_group:
            merged_boxes.append(self.merge_group(current_group))
        
        return merged_boxes

    def merge_group(self, group: list[dict]) -> dict:
        x_min = min(box['bbox'][0][0] for box in group)
        y_min = min(box['bbox'][0][1] for box in group)
        x_max = max(box['bbox'][2][0] for box in group)
        y_max = max(box['bbox'][2][1] for box in group)
        
        # merged_text = ' '.join([box['text'] for box in group])
        return {
            'bbox': [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            # 'text': merged_text,
            # 'prob': min(box['prob'] for box in group) 
        }

    def crop_image(self, image: Image.Image, bbox: list[list[int]]) -> Image.Image:
        # x1, y1 = bbox[0]
        # x2, y2 = bbox[2]
        # return image.crop((x1, y1, x2, y2))

        image = np.array(image)
        src_pts = np.array(bbox, dtype="float32")

        # Compute width and height of the new image after cropping
        width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
        height = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))

        # Define the destination points for the perspective transform
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply the perspective transformation to get the cropped image
        warped = cv2.warpPerspective(image, M, (width, height))

        # Convert back to PIL image
        cropped_pil_image = Image.fromarray(warped)

        return cropped_pil_image
