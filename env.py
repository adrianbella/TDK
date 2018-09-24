import random
import imageio
import glob
import cv2
import numpy as np
import math


class Environment:
    traning_names = np.array(
        ['andromeda', 'assistant', 'dreyar', 'eve', 'jasper', 'kachujin', 'liam', 'lola', 'malcolm', 'mark', 'medea',
         'peasant'])
    validation_names = np.array(['regina', 'remy', 'stefani'], dtype='U10')

    img_array = np.zeros((7, 45, 21, 200, 200))

    current_name = ''
    current_state = np.zeros(3)

    actions = np.array(['forward', 'backward', 'up', 'down', 'left', 'right'])

    def __init__(self):
        self.read_files()

    def read_files(self):

        self.current_name = self.get_random_name()
        current_figure_screenShots_path = "./Screenshots/" + self.current_name + "/*.png"

        for im_path in glob.glob(current_figure_screenShots_path):
            img = imageio.imread(im_path)
            gray_scale_img = self.gray_scale(img)

            x, y, z = self.get_descartes_coordinates(im_path)
            r, fi, theta = self.get_polar_coordinates(x, y, z)

            r_index = int((r - 0.5) / 0.15)  # r_index [0:6]
            fi_index = int(fi / 8)  # fi_index [0:44]
            theta_index = int((theta - 10) / 8)  # theta_index [0:20]

            self.img_array[r_index, fi_index, theta_index] = gray_scale_img

    def get_random_name(self):
        figure_index = random.randint(0, len(self.traning_names) - 1)
        name = self.traning_names[figure_index]
        self.traning_names = np.delete(self.traning_names, figure_index)
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

        # convert fi to [0:360] interval
        if fi < 0:
            fi += 360

        if z > 0:
            if x > 0:
                fi = 180 - fi
            elif x < 0:
                fi = 180 + (360 - fi)
        # -------------------------------

        return r, fi, theta

    def get_random_initial_state(self):
        r_index = random.randint(0, 6)
        fi_index = random.randint(0, 44)
        theta_index = random.randint(0, 20)

        self.current_state = np.array([r_index, fi_index, theta_index])

        return self.img_array[r_index, fi_index, theta_index]

    def step(self, action_index):
        action = self.actions[action_index]

        previous_state = self.current_state

        r_index = self.current_state[0]
        fi_index = self.current_state[1]
        theta_index = self.current_state[2]

        if action == 'forward':
            if r_index != 0:  # check if we can move forward
                r_index -= 1
            print('forward')
        elif action == 'backward':
            if r_index != 6:  # check if we can move backward
                r_index += 1
            print('backward')
        elif action == 'up':
            if theta_index != 0:  # check if we can move up
                theta_index -= 1
            print('up')
        elif action == 'down':
            if theta_index != 20:  # check if we can move down
                theta_index += 1
            print('down')
        elif action == 'left':
            if fi_index == 0:  # we are across from the figure
                fi_index = 44
            else:  # all other cases
                fi_index -= 1
            print('left')
        elif action == 'right':
            if fi_index == 44:
                fi_index = 0
            else:
                fi_index += 1
            print('right')

        self.current_state = np.array(
            [r_index, fi_index, theta_index])  # save the polar coordinates of the current state

        reward = self.get_reward(previous_state)  # calcuate the reward
        observation = self.img_array[r_index, fi_index, theta_index]  # find the new camera input

        print(previous_state)
        print(self.current_state)

        return reward, observation

    def get_reward(self, prev_state):
        return random.randint(0, 1)
