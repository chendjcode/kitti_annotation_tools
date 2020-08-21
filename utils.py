import numpy as np
import open3d as o3d
import cv2 as cv
import os
from typing import Tuple, List

lines = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [4, 5],
    [4, 6],
    [5, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def bin2points(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def points2cloud(_points: np.ndarray) -> o3d.geometry.PointCloud:
    points = np.copy(_points)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    # c = points[:, 3:4] / points[:, 3:4].max()
    # cloud.colors = o3d.utility.Vector3dVector(
    #     np.hstack([c, c, c]))
    return cloud


def points_split(_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.copy(_points)
    x_min, y_min, z_min, i_min = points.min(0)
    x_max, y_max, z_max, i_max = points.max(0)
    step = 1.0
    ground_points = []
    object_points = []
    for x in np.arange(x_min, x_max, step):
        for y in np.arange(y_min, y_max, step):
            condition = (points[:, 0] >= x) & (points[:, 0] < x + step) & \
                        (points[:, 1] >= y) & (points[:, 1] < y + step)
            cell_points: np.ndarray = points[condition]
            if cell_points.size == 0:
                continue
            height = (cell_points.max(0) - cell_points.min(0))[2]
            if height < step * 0.22:
                ground_points.append(cell_points)
            else:
                object_points.append(cell_points)
            # speed up
            condition = np.logical_not(condition)
            points = points[condition]
    ground_points = np.vstack(ground_points)
    object_points = np.vstack(object_points)
    return ground_points, object_points


def ground_segment(_points: np.ndarray) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, np.ndarray]:
    points = np.copy(_points)
    above = points[points[:, 2] >= 0.0]
    below = points[points[:, 2] < 0.0]
    cloud = points2cloud(below)
    plane_model, indices = cloud.segment_plane(distance_threshold=0.1, ransac_n=below.shape[0], num_iterations=100)
    ground_cloud = cloud.select_by_index(indices)
    non_ground_cloud = cloud.select_by_index(indices, invert=True) + points2cloud(above)
    return non_ground_cloud, ground_cloud, plane_model


def get_angle_between_points2d(vector_1, vector_2) -> float:
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def rotate_points(_points: np.ndarray, rotation_xyz: tuple) -> np.ndarray:
    points = np.copy(_points)
    cloud = points2cloud(points[:, 0:3])
    R = cloud.get_rotation_matrix_from_xyz(rotation_xyz)
    cloud.rotate(R, (0, 0, 0))
    return np.asarray(cloud.points)


def gen_bev(points: np.ndarray, rotation_y=0.0, radius=30.0, resolution=0.05) -> np.ndarray:
    points = rotate_points(points, (0, 0, rotation_y))
    points[:, 1] += radius
    points[:, 0] = radius - points[:, 0]
    points[:, 1] = radius * 2 - points[:, 1]
    points[:, 0:2] /= resolution
    points = points.astype(np.int)
    bev_img = np.zeros(
        (int(radius / resolution), int(radius / resolution * 2), 3))  # height, width, channels
    for x, y in points[:, 0:2]:
        if 0 <= x < bev_img.shape[0] and 0 <= y < bev_img.shape[1]:
            bev_img[x, y] += 1
    return bev_img


def add_bbox3d_bev(bev_img: np.ndarray, _bbox3d_bevs: List[np.ndarray], rotation_y: float,
                   radius: float = 30.0, resolution: float = 0.05):
    bbox3d_bevs = np.copy(_bbox3d_bevs)
    for bbox3d_bev in bbox3d_bevs:
        bbox3d_bev[:, 1] += radius
        bbox3d_bev[:, 0] = radius - bbox3d_bev[:, 0]
        bbox3d_bev[:, 1] = radius * 2 - bbox3d_bev[:, 1]
        bbox3d_bev[:, 0:2] /= resolution
        bbox3d_bev = bbox3d_bev.astype(np.int)
        bbox3d_bev = [rotate2d(tuple(point), rotation_y) for point in bbox3d_bev]
        for i in range(len(bbox3d_bev)):
            cv.line(bev_img, tuple(bbox3d_bev[i]), tuple(bbox3d_bev[(i + 1) % len(bbox3d_bev)]), (0, 255, 0), 1)


def roll_back_bbox3d_bev_coord(left_up_x: int, left_up_y: int, right_down_x: int, right_down_y: int,
                               rotation_y: float, radius: float = 30.0, resolution: float = 0.05):
    four_corns = np.asarray([
        (left_up_x, left_up_y), (left_up_x, right_down_y), (right_down_x, right_down_y), (right_down_x, left_up_y)],
        dtype=np.float)
    four_corns *= resolution
    four_corns[:, 0] = radius - four_corns[:, 0]
    four_corns[:, 1] = radius * 2 - four_corns[:, 1]
    four_corns[:, 1] -= radius
    for i in range(len(four_corns)):
        four_corns[i] = rotate2d(four_corns[i], -rotation_y)
    return four_corns


def rotate2d(point: tuple, theta: float) -> tuple:
    x, y = point
    c, s = np.cos(theta), np.sin(theta)
    return int(x * c - y * s), int(x * s + y * c)


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def get_bbox3d(dimensions, location, rotation_y):
    h, w, l = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    x, y, z = float(location[0]), float(location[1]), float(location[2])
    # return [
    #     [x - width / 2, y - height, z - length / 2],
    #     [x + width / 2, y - height, z - length / 2],
    #     [x - width / 2, y, z - length / 2],
    #     [x + width / 2, y, z - length / 2],
    #     [x - width / 2, y - height, z + length / 2],
    #     [x + width / 2, y - height, z + length / 2],
    #     [x - width / 2, y, z + length / 2],
    #     [x + width / 2, y, z + length / 2]
    # ]
    # 3d bounding box corners
    # x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    # y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    # z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    x_corners = [-l / 2, l / 2, -l / 2, l / 2, -l / 2, l / 2, -l / 2, l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    # rotate and translate 3d bounding box
    R = roty(rotation_y)
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    return corners_3d.T


def get_label(path):
    with open(path, "r") as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        line = line.split(" ")
        label = {
            "bbox2d": np.asarray(line[4:8], dtype=np.float32),
            "bbox3d": np.asarray(get_bbox3d(line[8:11], line[11:14], float(line[14])), dtype=np.float32),
        }
        labels.append(label)
    return labels


def draw_bbox3d_on_cloud(points, labels, calib):
    bbox3ds = []
    for label in labels:
        corns = cam2vel(label, calib)
        colors = [[0, 1, 0] for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corns),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbox3ds.append(line_set)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    bbox3ds.append(pcd)
    o3d.visualization.draw_geometries(bbox3ds)


def get_calib(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    lines.pop(-1)
    P2 = np.asarray(lines[2].split(" ")[1:], dtype=np.float32).reshape(3, 4)
    P2 = np.concatenate((P2, np.asarray([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    R0_rect = np.asarray(lines[4].split(" ")[1:], dtype=np.float32).reshape(3, 3)
    R0_rect = np.concatenate((R0_rect, np.asarray([[0, 0, 0]], dtype=np.float32)), axis=0)
    R0_rect = np.concatenate((R0_rect, np.asarray([[0], [0], [0], [1]], dtype=np.float32)), axis=1)
    Tr_velo_to_cam = np.asarray(lines[5].split(" ")[1:], dtype=np.float32).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate((Tr_velo_to_cam, np.asarray([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    calib = {
        "P2": P2,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam
    }
    return calib


def get_img(path):
    return cv.imread(path)


def draw_bbox2d_on_img(img, labels):
    for label in labels:
        bbox2d = label["bbox2d"]
        cv.rectangle(img, (bbox2d[0], bbox2d[1]), (bbox2d[2], bbox2d[3]), color=(255, 0, 0), thickness=3)
    return img


def cam2vel(label, calib):
    Tr = np.linalg.inv(calib["Tr_velo_to_cam"] @ calib["R0_rect"])
    points = np.concatenate((label["bbox3d"], np.ones((label["bbox3d"].shape[0], 1))), axis=1)
    points = Tr @ points.T
    return points.T[:, :3]


def draw_bbox3d_on_img(img, labels, calib):
    for label in labels:
        corns = calib["P2"] @ np.concatenate((label["bbox3d"], np.ones((label["bbox3d"].shape[0], 1))), axis=1).T
        corns = corns.T[:, :2] / corns.T[:, 2:3]
        corns = corns.astype(np.int)
        for line in lines:
            img = cv.line(img, tuple(corns[line[0]]), tuple(corns[line[1]]), (0, 255, 0), 2)
    return img


def main():
    points_path = os.path.join(dataDir, "velodyne", idx + ".bin")
    points = bin2points(points_path)[:, :3]
    calib_path = os.path.join(dataDir, "calib", idx + ".txt")
    calib = get_calib(calib_path)
    proposal_dir = "/home/cdj/avod_Re/avod/data/outputs/pyramid_pedestrian_example/predictions/images_2d/predictions/train/120000/0.5"
    proposal_path = os.path.join(proposal_dir, idx + ".txt")
    label_path = os.path.join(dataDir, "label_2", idx + ".txt")
    labels = get_label(proposal_path)
    img_path = os.path.join(dataDir, "image_2", idx + ".png")
    img = get_img(img_path)

    img = draw_bbox2d_on_img(img, labels)
    img = draw_bbox3d_on_img(img, labels, calib)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', img)
    cv.waitKey(0)

    draw_bbox3d_on_cloud(points, labels, calib)

    cv.destroyAllWindows()


if __name__ == '__main__':
    dataDir = "/home/cdj/Kitti/object/training"
    idx = "000053"
    main()
