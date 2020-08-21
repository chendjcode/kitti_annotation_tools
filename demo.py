import open3d as o3d
import numpy as np
import utils


def main():
    bin_path = "/home/cdj/Kitti/object_bak/training/velodyne/000061.bin"
    points = utils.bin2points(bin_path)
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
            # condition = np.logical_not(condition)
            # points = points[condition]
    ground_points = np.vstack(ground_points)
    object_points = np.vstack(object_points)





if __name__ == '__main__':
    main()
