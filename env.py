import imageio
import glob
import cv2
import numpy as np
import math


class Environment:
    all_screenShots_path = "./Screenshots/*/*.png"
    test_screenShots_path = "./Screenshots/andromeda/*.png"
    img_array = np.array(99240)

    def __init__(self):
        self.read_files()

    def read_files(self):
        for im_path in glob.glob(self.all_screenShots_path):
            img = imageio.imread(im_path)

            name = self.get_name(im_path)

            gray_scale_img = self.gray_scale(img)

            x, y, z = self.get_descartes_coordinates(im_path)
            r, fi, theta = self.get_polar_coordinates(x, y, z)

            element = np.array([(name, r, fi, theta, )],
                               dtype=[('name', 'U10'), ('radius', 'f3'), ('fi_angel', 'f3'), ('theta_angel', 'f3')])

    def get_name(self, img_path):
        first_cut = img_path[14:]
        name = first_cut.split('/')[0]
        return name

    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def get_descartes_coordinates(self, img_path):
        from_str = img_path.split('(')[1]
        coordinates_with_coma = from_str.split(')')[0]
        str_x, str_y, str_z = coordinates_with_coma.split(',')

        return float(str_x), float(str_y), float(str_z)

    def get_polar_coordinates(self, x, y, z):
        shifted_y = y - 1.6
        r = math.sqrt(math.pow(x, 2) + math.pow(shifted_y, 2) + math.pow(z, 2))
        theta = math.acos(shifted_y / r)
        fi = math.asin(x / (math.sin(theta) * r))

        # convert theta an fi from rad to degree
        theta = math.degrees(theta)
        fi = math.degrees(fi)

        return r, fi, theta
