import numpy as np
import open3d as o3d
import cv2 as cv
from PIL import Image
from typing import Tuple
import utils
import os

boundary = {
    "minX": -30.0,
    "maxX": 30.0,
    "minY": -30.0,
    "maxY": 30.0,
    "minZ": -3.0,
    "maxZ": 3.0,
}


def remove_points(_points: np.ndarray) -> np.ndarray:
    points = _points.copy()
    mask = np.where((points[:, 0] >= boundary["minX"]) & (points[:, 0] <= boundary["maxX"]) &
                    (points[:, 1] >= boundary["minY"]) & (points[:, 1] <= boundary["maxY"]) &
                    (points[:, 2] >= boundary["minZ"]) & (points[:, 2] <= boundary["maxZ"]))
    points = points[mask]
    return points


def make_bev(points_: np.ndarray, resolution: float = 0.1, lines: int = 16) -> np.ndarray:
    points = points_.copy()
    points = remove_points(points)

    bev_width = np.int_(np.floor(boundary["maxY"] - boundary["minY"]) / resolution)
    bev_hight = np.int_(np.floor(boundary["maxX"] - boundary["minX"]) / resolution)

    points[:, 0] = np.int_(np.floor(points[:, 0] / resolution) + bev_hight / 2)  # TODO
    points[:, 1] = np.int_(np.floor(points[:, 1] / resolution) + bev_width / 2)

    indices = np.lexsort((-points[:, 2], points[:, 1], points[:, 0]))
    points = points[indices]

    # _ = the sorted unique values
    _, indices, counts = np.unique(points[:, 0:2], axis=0, return_index=True, return_counts=True)
    points = points[indices]

    density_map = np.zeros((bev_hight, bev_width))
    density_map[np.int_(points[:, 0]), np.int_(points)[:, 1]] = np.minimum(1.0, (counts / lines) * resolution)

    intensity_map = np.zeros((bev_hight, bev_width))
    intensity_map[np.int_(points[:, 0]), np.int_(points)[:, 1]] = points[:, 3]

    height_map = np.zeros((bev_hight, bev_width))
    height_map[np.int_(points[:, 0]), np.int_(points)[:, 1]] \
        = np.where(points[:, 2] > 0, points[:, 2] / boundary["maxZ"], points[:, 2] / boundary["minZ"])

    bgr_map = np.zeros((3, bev_hight, bev_width))
    bgr_map[0, :, :] = intensity_map
    bgr_map[1, :, :] = height_map
    bgr_map[2, :, :] = density_map
    bgr_map = np.moveaxis(bgr_map, 0, -1)
    bgr_map *= 255
    return np.ascontiguousarray(bgr_map.astype(np.uint8))


drawing = False
left_top = (-1, -1)
right_bottom = (-1, -1)


def draw_bbox2d_callback(event: int, x: int, y: int, flags: int, param):
    global drawing, left_top, right_bottom, bev
    bev_canvas = bev.copy()
    if event == cv.EVENT_LBUTTONDBLCLK:
        if not drawing:
            drawing = True
            left_top = x, y
        else:
            drawing = False
            cv.rectangle(bev_canvas, left_top, right_bottom, (255, 0, 0), 1)
            bev = bev_canvas
            cv.imshow("bev", bev)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.rectangle(bev_canvas, left_top, (x, y), (255, 0, 0), 1)
            cv.imshow("bev", bev_canvas)
            right_bottom = x, y


bev = None


def annotation():
    global bev
    bin_dir = "/home/cdj/Kitti/object/training/velodyne"
    bin_path = os.path.join(bin_dir, "000050.bin")
    points_ = utils.bin2points(bin_path)
    points_ = remove_points(points_)

    cv.namedWindow("bev", cv.WINDOW_NORMAL)
    cv.resizeWindow("bev", 600, 600)
    cv.setMouseCallback("bev", draw_bbox2d_callback)

    theta = np.pi
    while True:
        rotation = utils.rotz(theta)
        points = points_.copy()
        points[:, 0:3] = (rotation @ points[:, 0:3].T).T
        bev = make_bev(points)
        cv.imshow("bev", bev)
        key = cv.waitKey(0) & 0xFF
        if key == 27:
            break
        elif key == ord('a'):
            theta += np.pi / 30
        elif key == ord('d'):
            theta -= np.pi / 30
        elif key == ord('s'):
            theta = np.pi

    cv.destroyAllWindows()


def ground_remove_bak(_points: np.ndarray, grid_size: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    points = _points.copy()

    points[:, 0] = np.int_(np.floor(points[:, 0] / grid_size))
    points[:, 1] = np.int_(np.floor(points[:, 1] / grid_size))
    indices = np.lexsort((-points[:, 2], points[:, 1], points[:, 0]))
    points = points[indices]
    # _ = the sorted unique values
    unique, indices, counts = np.unique(points[:, 0:2], axis=0, return_index=True, return_counts=True)
    ground = []
    objects = []
    h_hat = 0.0
    s = 0.09
    for x, y in unique:
        mask = np.where((points[:, 0] == x) & (points[:, 1] == y))
        grid = points[mask]
        H = grid.max(0)[2]
        h = grid.min(0)[2]
        if ((H - h) < s) and (H < (h_hat + s)):  # ground
            ground.append(grid)
        else:  # objects
            h_hat = H
            objects.append(grid)
    ground = np.vstack(ground)
    objects = np.vstack(objects)
    return objects, ground


def ground_remove(_points: np.ndarray, grid_size: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    points = _points.copy()
    ground = []
    above = []
    s = grid_size * 0.22
    for x in np.arange(points.min(0)[0], points.max(0)[0], grid_size):
        for y in np.arange(points.min(0)[1], points.max(0)[1], grid_size):
            voxel = points[(points[:, 0] >= x) & (points[:, 0] < x + grid_size) &
                           (points[:, 1] >= y) & (points[:, 1] < y + grid_size)]
            if voxel.shape[0] != 0:
                H = voxel.max(0)[2]
                h = voxel.min(0)[2]
                if (H - h) > s:
                    above.append(voxel)
                else:
                    ground.append(voxel)
    return np.vstack(above), np.vstack(ground)


def main():
    bin_dir = "/home/cdj/Kitti_o/object/training/velodyne"
    bin_path = os.path.join(bin_dir, "000050.bin")
    _points = utils.bin2points(bin_path)
    points = remove_points(_points)
    above, ground = ground_remove(points, grid_size=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground[:, :3])
    # pcd.segment_plane()
    # o3d.visualization.draw_geometries([pcd])

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=ground.shape[0], num_iterations=1)
    print(plane_model)
    # -2.047197e-02 -9.997891e-01 -1.656901e-03 1.629543e+00


def call_back(vis: o3d.visualization.VisualizerWithVertexSelection):
    for point in vis.get_picked_points():
        print(point.coord, point.index)


def remove_kitti_ground():
    bin_dir = "/home/cdj/Kitti/object/training/velodyne"
    for bin in os.listdir(bin_dir):
        bin_path = os.path.join(bin_dir, bin)
        points = utils.bin2points(bin_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # o3d.visualization.draw_geometries([pcd])
        # o3d.visualization.draw_geometries_with_vertex_selection([pcd])
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window()
        vis.add_geometry(pcd)

        vis.register_selection_changed_callback(lambda: call_back(vis))

        vis.run()
        # while True:
        #     # vis.update_geometry(pcd)
        #     vis.poll_events()
        #     vis.update_renderer()
        vis.destroy_window()
        break
        above, _ = ground_remove(points, grid_size=1.0)
        above.tofile(bin_path)
        print(bin_path)


if __name__ == '__main__':
    remove_kitti_ground()
