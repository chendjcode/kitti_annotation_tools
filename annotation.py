import open3d as o3d
import cv2 as cv
import numpy as np
import utils
from functools import partial


class AnnotationFromBev:
    def __init__(self, bin_path, radius=30.0):
        self.radius = radius
        self.points = utils.bin2points(bin_path)
        self.range_crop()
        self.rotation_y = 0.0
        self.bev_img = utils.gen_bev(self.points, rotation_y=0)
        self.bbox3d_bevs = []
        self.drawing = False
        self.left_up_x, self.left_up_y = -1, -1

    def range_crop(self):
        self.points = self.points[self.points[:, 0] >= 0]
        self.points = self.points[np.linalg.norm(self.points[:, 0:2], axis=1) <= self.radius]

    def draw_rectangle(self, event: int, x: int, y: int, flags: int, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.drawing = True
                self.left_up_x, self.left_up_y = x, y
            else:
                cv.rectangle(self.bev_img, (self.left_up_x, self.left_up_y), (x, y), (0, 255, 0), 1)
                self.bbox3d_bevs.append(
                    utils.roll_back_bbox3d_bev_coord(self.left_up_x, self.left_up_y, x, y, self.rotation_y))
                cv.imshow("bev", self.bev_img)
                self.drawing = False
        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing:
                bev_img_tmp = self.bev_img.copy()
                cv.rectangle(bev_img_tmp, (self.left_up_x, self.left_up_y), (x, y), (0, 255, 0), 1)
                cv.imshow("bev", bev_img_tmp)

    def annotate(self):
        while True:
            self.bev_img = utils.gen_bev(self.points, rotation_y=self.rotation_y)
            utils.add_bbox3d_bev(self.bev_img, self.bbox3d_bevs, self.rotation_y)
            print(self.bbox3d_bevs)
            cv.imshow("bev", self.bev_img)
            cv.setMouseCallback("bev", self.draw_rectangle, param="test")
            k = cv.waitKey(0) & 0xFF
            print(k)
            if k == 32:
                break
            elif k == ord('a'):
                self.rotation_y += np.pi / 9
            elif k == ord('d'):
                self.rotation_y -= np.pi / 9
            elif k == ord('r'):
                self.rotation_y = 0


class AnnotationFrom3d:
    def __init__(self, bin_path, img_path, label_path):
        self.points = utils.bin2points(bin_path)
        self.img = cv.imread(img_path)
        non_ground_cloud, ground_cloud, plane_model = utils.ground_segment(self.points)
        self.points = np.asarray(non_ground_cloud.points)
        self.ground_model = plane_model
        self.range_crop()
        self.K = o3d.camera.PinholeCameraIntrinsic(
            width=1280, height=720, fx=1981.69556, cx=623.69778, fy=2094.65771, cy=435.08522)
        self.T: np.ndarray = np.fromstring("0.14923799 -0.9886232 0.01876693 -0.05842154 " +
                                           "-0.0686831 -0.02929803 -0.9972083 -0.3735077 " +
                                           "0.986413 0.14753239 -0.07227408 -2.1367278 0 0 0 1",
                                           dtype=float, sep=' ').reshape((4, 4))
        self.selected_vertex = []
        self.bbox3d = {}
        self.bbox2d = ()
        self.label_path = label_path

    def range_crop(self, radius=30.0):
        self.points = self.points[self.points[:, 0] > 0]
        self.points = self.points[np.linalg.norm(self.points[:, 0:2], axis=1) <= radius]

    def selection_callback(self, vis: o3d.visualization.VisualizerWithVertexSelection):
        self.selected_vertex.clear()
        for point in vis.get_picked_points():
            self.selected_vertex.append(point.coord)

    def get_axis_aligned_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        obj_points = np.vstack(self.selected_vertex)
        obj_cloud = utils.points2cloud(obj_points)
        bounding_box: o3d.geometry.AxisAlignedBoundingBox = obj_cloud.get_axis_aligned_bounding_box()
        bounding_box.color = (0, 1, 0)
        return bounding_box

    def add_bottom(self):
        obj_points = np.vstack(self.selected_vertex)
        center = (obj_points.min(0) + obj_points.max(0)) / 2
        bottom_center = (center[0], center[1],
                         (-1 * self.ground_model[3] - self.ground_model[0] * center[0] - self.ground_model[1] * center[
                             1]) / self.ground_model[2])
        self.selected_vertex.append(bottom_center)

    def annotate3d(self):
        cloud = utils.points2cloud(self.points)
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.register_selection_changed_callback(lambda: self.selection_callback(vis))
        vis.create_window(width=self.K.width, height=self.K.height)
        ctr: o3d.visualization.ViewControl = vis.get_view_control()
        vis.add_geometry(cloud)
        pinhole_camera_parameters: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
        pinhole_camera_parameters.extrinsic = self.T
        ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters)
        vis.run()
        self.add_bottom()  # until ground
        self.gen_bbox3d()

    def gen_bbox3d(self):
        bounding_box3d_cloud = utils.points2cloud(np.vstack(self.selected_vertex))
        x, y, z = bounding_box3d_cloud.get_max_bound() - bounding_box3d_cloud.get_min_bound()
        self.bbox3d["dimensions"] = z, x, y  # height, width, length
        bounding_box3d_cloud.transform(self.T)
        bounding_box3d: o3d.geometry.AxisAlignedBoundingBox = bounding_box3d_cloud.get_axis_aligned_bounding_box()
        cloud = utils.points2cloud(self.points)
        cloud.transform(self.T)
        self.bbox3d["alpha"] = utils.get_angle_between_points2d(np.asarray([0, 1]), np.asarray(
            [bounding_box3d.get_center()[1], bounding_box3d.get_center()[2]]))
        center = (bounding_box3d.get_min_bound() + bounding_box3d.get_max_bound()) / 2
        self.bbox3d["location"] = center[0], center[1] + z / 2, center[2]

    def show_annotation3d(self):
        cloud = utils.points2cloud(self.points)
        bbox3d = self.get_axis_aligned_bounding_box()
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.K.width, height=self.K.height)
        ctr: o3d.visualization.ViewControl = vis.get_view_control()
        vis.add_geometry(cloud)
        vis.add_geometry(bbox3d)
        pinhole_camera_parameters: o3d.camera.PinholeCameraParameters = ctr.convert_to_pinhole_camera_parameters()
        pinhole_camera_parameters.extrinsic = self.T
        ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters)
        vis.run()

    def draw_bbox2d(self, event: int, x: int, y: int, flags: int, param: dict):
        if event == cv.EVENT_LBUTTONDOWN:
            if not param["drawing"]:
                param["drawing"] = True
                param["left_up"] = x, y
            else:
                param["right_down"] = x, y
                cv.rectangle(param["img"], param["left_up"], param["right_down"], (0, 255, 0), 1)
                cv.imshow("img", param["img"])
                param["drawing"] = False
        elif event == cv.EVENT_MOUSEMOVE:
            if param["drawing"]:
                img_tmp = np.copy(param["img"])
                cv.rectangle(img_tmp, param["left_up"], (x, y), (0, 255, 0), 1)
                cv.imshow("img", img_tmp)

    def annotate2d(self):
        bbox2d = {
            "img": self.img,
            "drawing": False,
            "left_up": [-1, -1],
            "right_down": [-1, -1]
        }
        cv.imshow("img", self.img)
        cv.setMouseCallback("img", self.draw_bbox2d, param=bbox2d)
        cv.waitKey(0)
        self.bbox2d = *bbox2d["left_up"], *bbox2d["right_down"]

    def annotate(self):
        self.annotate3d()
        self.show_annotation3d()
        self.annotate2d()
        return "Pedestrian " + "0 " + "0 " + str(self.bbox3d["alpha"]) + " " + " ".join(map(str, self.bbox2d)) + " " \
               + " ".join(map(str, self.bbox3d["dimensions"])) + " " + \
               " ".join(map(str, self.bbox3d["location"])) + " 0 " + "1"

    def write_label(self):
        label = self.annotate()
        with open(self.label_path, "w") as f:
            f.write(label)


def main():
    bin_path = "/home/cdj/Kitti/object/training/velodyne/000061.bin"
    img_path = "/home/cdj/Kitti/object/training/image_2/000061.png"
    label_path = "/home/cdj/Kitti/object/training/label_2/000061.txt"
    an = AnnotationFrom3d(bin_path, img_path, label_path)
    an.write_label()


if __name__ == '__main__':
    main()
