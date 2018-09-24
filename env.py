import imageio
import glob
import cv2
import numpy as np
import math


class Environment:
    all_screenShots_path = "./Screenshots/*/*.png"
    test_screenShots_path = "./Screenshots/andromeda/*.png"
    img_array = np.zeros(6615, dtype=[('name', 'U10'), ('radius', 'f4'), ('fi_angel', 'f4'), ('theta_angel', 'f4'),
                                      ('img', 'i4', (200, 200))])
    fi_array = np.zeros(100, dtype=int)

    def __init__(self):
        self.read_files()

    def read_files(self):
        idx = 0
        for im_path in glob.glob(self.test_screenShots_path):
            img = imageio.imread(im_path)

            name = self.get_name(im_path)

            gray_scale_img = self.gray_scale(img)

            x, y, z = self.get_descartes_coordinates(im_path)
            r, fi, theta = self.get_polar_coordinates(x, y, z)

            r_index = int((r - 0.5) / 0.15)  # r_index [0:6]
            fi_index = int(fi / 8)  # fi_index [0:44]
            theta_index = int((theta - 10) / 8)  # theta_index [0:20]

            print(r, r_index, fi, fi_index, theta, theta_index)

            element = np.array([(name, r, fi_index, theta_index, gray_scale_img)],
                               dtype=[('name', 'U10'), ('radius_index', 'i2'), ('fi_angel_index', 'i2'), ('theta_angel_index', 'i2'),
                                      ('img', 'i4', (200, 200))])

            self.img_array[idx] = element
            idx += 1

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

        theta = int(round(theta))
        fi = int(round(fi))
        r = round(r, 2)

        # convert fi to [0:360] intervall
        if fi < 0:
            fi += 360

        if z > 0:
            if x > 0:
                fi = 180 - fi
            elif x < 0:
                fi = 180 + (360 - fi)
        # -------------------------------

        return r, fi, theta
