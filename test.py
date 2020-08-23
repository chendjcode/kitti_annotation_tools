import open3d as o3d
import numpy as np
import utils


def main():
    bin_path = "/home/cdj/Kitti/object/training/velodyne/000061.bin"
    points = utils.bin2points(bin_path)
    o3d.visualization.draw_geometries([utils.points2cloud(points)])
    # ground_points, object_points = utils.points_split(points)
    # o3d.visualization.draw_geometries([utils.points2cloud(ground_points), utils.points2cloud(object_points)])





if __name__ == '__main__':
    main()
