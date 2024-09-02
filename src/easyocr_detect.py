import easyocr
from PIL import Image
import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import distance as dist


class EasyOCRDetect:
  def __init__(self):
        self.reader = easyocr.Reader(['en'])

  def extract_bounding_boxes(self, image: Image.Image) -> list[dict]:
      image_np = np.array(image)
      if image_np.dtype != np.uint8:
          image_np = image_np.astype(np.uint8)

      results = self.reader.readtext(image_np)
      bounding_boxes = []

      for (bbox, _, _) in results:
        bounding_boxes.append(bbox)

      try:
        merged_boxes = []
        bbox_grps = self.get_box_groups(bounding_boxes)
        for bbox_grp in bbox_grps:
          merged_boxes.append(self.get_merged_box(bbox_grp))
          # merged_boxes.append(self.scale_rotate_coordinates(self.get_merged_box(bbox_grp), 0.95, 0.9))
        return merged_boxes

      except Exception as e:
        print(f"Error : {e}")
        return bounding_boxes


  def cart2pol(self, x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


  def pol2cart(self, theta, rho):
      x = rho * np.cos(theta)
      y = rho * np.sin(theta)
      return x, y

  def is_intersecting_coordinates(self, coords1, coords2):
      isIntersect = False
      ret,_ = cv2.rotatedRectangleIntersection(cv2.minAreaRect(coords1), cv2.minAreaRect(coords2) )
      if ret != 0:
          isIntersect = True
          # print("intersecting")
      return isIntersect

  def scale_rotate_coordinates(self, bbox: list, fw: float, fh: float) -> list:
      bbox_np = np.array(bbox)
      centroid = np.mean(bbox_np, axis=0)

      vec_width = bbox_np[1] - bbox_np[0]  # vector from top-left to top-right
      vec_height = bbox_np[3] - bbox_np[0] # vector from top-left to bottom-left

      vec_width_scaled = vec_width * fw
      vec_height_scaled = vec_height * fh

      new_bbox = [
          centroid - 0.5 * (vec_width_scaled + vec_height_scaled),
          centroid + 0.5 * (vec_width_scaled - vec_height_scaled),
          centroid + 0.5 * (vec_width_scaled + vec_height_scaled),
          centroid - 0.5 * (vec_width_scaled - vec_height_scaled)
      ]

      return np.array([list(map(int, point)) for point in new_bbox] ,  dtype=np.float32)

  def is_mergable(self, box1, box2):
        is_merge = False
        coords1_scaled = self.scale_rotate_coordinates(box1, 1.25, 0.5)
        coords2_scaled = self.scale_rotate_coordinates(box2, 1.25, 0.5)
        # print(f"COORDS1: {coords1} \nand stretched: {coords1_scaled}")
        if self.is_intersecting_coordinates(coords1=coords1_scaled, coords2=coords2_scaled):
            is_merge = True

        return is_merge

  def get_box_groups(self, boxList):
      box_group_list = [] #[[box1,box2,box3],[box4,box5],[box6,box7,box8,box9]]
      box_matrix = np.zeros([len(boxList),len(boxList)])
      for i, box1 in enumerate(boxList):
          for j, box2 in enumerate(boxList):
              if j > i:
                  if self.is_mergable(box1, box2):
                      box_matrix[i][j] = 1
                  box_matrix[j][i] = box_matrix[i][j]
      #get box group list
      n_components, labels = connected_components(box_matrix)
      # print(f"number_components: {n_components}\nLABELS: {labels}")
      for i in range(n_components):
          indxs = np.where(labels == i)[0]
          # print(f"indices for label {i} is: {indxs}")
          bgroup = [boxList[idx] for idx in indxs]
          # print(f"groups: {bgroup}")
          box_group_list.append(bgroup)
      return box_group_list

  def order_points(self, pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

  def get_merged_box(self, box_group):
    merged_box = box_group[0]
    if len(box_group) > 1:
        cnt = []
        for box in box_group:
            cnt.extend(box)
        cnt = np.array([[list(pt)] for pt in cnt],  dtype=np.float32)
        merged_box = np.array(self.order_points(np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))), dtype=np.float32)
    return merged_box

  def crop_image(self, image: Image.Image, bbox: list[list[int]]) -> Image.Image:
      image = np.array(image)
      src_pts = np.array(bbox, dtype="float32")

      width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
      height = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))

      dst_pts = np.array([
          [0, 0],
          [width - 1, 0],
          [width - 1, height - 1],
          [0, height - 1]
      ], dtype="float32")

      M = cv2.getPerspectiveTransform(src_pts, dst_pts)

      warped = cv2.warpPerspective(image, M, (width, height))
      cropped_pil_image = Image.fromarray(warped)

      return cropped_pil_image