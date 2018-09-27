import random
import imageio
import glob
import cv2
import numpy as np
import math


class Environment:
    CLASS_TAG = 'Environment: '

    # set of the name of the figures were used during the training process
    traning_names = np.array(
        ['andromeda', 'assistant', 'dreyar', 'eve', 'jasper', 'kachujin', 'liam', 'lola', 'malcolm', 'mark', 'medea',
         'peasant'])

    # set of the name of the figures were used during the validation process
    validation_names = np.array(['regina', 'remy', 'stefani'], dtype='U10')  #

    img_array = np.zeros((7, 45, 21, 200, 200))

    current_name = ''
    current_state = np.zeros(3)

    actions = np.array(['forward', 'backward', 'up', 'down', 'left', 'right'])

    def __init__(self):
        try:
            self.read_files()
        except IOError:
            print(self.CLASS_TAG + "Error: can\'t find the files or read data")
        else:
            print(self.CLASS_TAG + "Reading screenshots was successful!")

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
        figures_count = len(self.traning_names)
        figure_index = random.randint(0, figures_count - 1)  # choose a random figure_index from the possible ones
        name = self.traning_names[figure_index]  # get the name of the figure
        self.traning_names = np.delete(self.traning_names,
                                       figure_index)  # delete the name from the list (to avoid repeated learning with the same one)
        return name

    def gray_scale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # TODO: cast to uint8

    def get_descartes_coordinates(self, img_path):

        # cut the required x, y and z string values from the filename
        from_str = img_path.split('(')[1]
        coordinates_with_coma = from_str.split(')')[0]
        str_x, str_y, str_z = coordinates_with_coma.split(',')
        # -----------------------------------------------------------

        return float(str_x), float(str_y), float(str_z)

    def get_polar_coordinates(self, x, y, z):

        shifted_y = y - 1.6

        r = math.sqrt(math.pow(x, 2) + math.pow(shifted_y, 2) + math.pow(z, 2))  # calculate the |r| value
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

        # validate action index
        if not (0 <= action_index <= 5):
            print(self.CLASS_TAG + 'Invalid action index: ' + action_index + ', It must be between [0:5]')
        # --------------------------------------------------------------------------------------------------

        action = self.actions[action_index]

        previous_state = self.current_state  # save the previous polar coordinates of the state

        r_index = self.current_state[0]
        fi_index = self.current_state[1]
        theta_index = self.current_state[2]

        if action == 'forward':
            if r_index != 0:  # check if we can move forward
                r_index -= 1
            print(self.CLASS_TAG + 'step forward')
        elif action == 'backward':
            if r_index != 6:  # check if we can move backward
                r_index += 1
            print(self.CLASS_TAG + 'step backward')
        elif action == 'up':
            if theta_index != 0:  # check if we can move up
                theta_index -= 1
            print(self.CLASS_TAG + 'up')
        elif action == 'down':
            if theta_index != 20:  # check if we can move down
                theta_index += 1
            print(self.CLASS_TAG + 'step down')
        elif action == 'left':
            if fi_index == 0:  # check if we are across from the figure
                fi_index = 44
            else:  # all other cases
                fi_index -= 1
            print(self.CLASS_TAG + 'step left')
        elif action == 'right':
            if fi_index == 44:  # check if we reached the maximal fi_index value
                fi_index = 0
            else:  # all other cases
                fi_index += 1
            print(self.CLASS_TAG + 'step right')

        # validate and set indexes
        if not (0 <= r_index <= 6):
            print(self.CLASS_TAG + 'Invalid r index: ' + r_index)
        elif not (0 <= fi_index <= 44):
            print(self.CLASS_TAG + 'Invalid fi index: ' + fi_index)
        elif not (0 <= theta_index <= 20):
            print(self.CLASS_TAG + 'Invalid theta index: ' + theta_index)
        else:
            self.current_state = np.array(
                [r_index, fi_index, theta_index])  # save the new polar coordinates of the current state
            observation = self.img_array[r_index, fi_index, theta_index]  # find the new camera input
            print('Previous position: ', previous_state)
            print('Current position: ', self.current_state)
        # -------------------------------------------------------------------------------------------------

        reward = self.get_reward(previous_state)  # calcuate the reward

        return reward, observation

    def get_reward(self, prev_state):  # TODO: megírni a jutalom függvényt
        return random.randint(0, 1)
