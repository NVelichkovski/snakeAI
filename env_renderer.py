# 42:36

from PIL import Image
import cv2
import numpy as np
from time import sleep
from matplotlib import pyplot as plt

from environment_generator import Environment
class EnvRenderer:
    def __init__(self, env: Environment):
        self.env = env
        self.img = None

    def render(self):
        self.img = Image.fromarray(self.env.matrix, 'RGB')
        self.img = self.img.resize((500, 500), Image.NONE)
        cv2.imshow('', np.array(self.img))
        cv2.waitKey(delay=0)