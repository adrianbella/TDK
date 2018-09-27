from PIL import Image
import numpy as np


class Display:

    @staticmethod
    def show_img(nump):

        img = Image.fromarray(nump)
        img.show()
