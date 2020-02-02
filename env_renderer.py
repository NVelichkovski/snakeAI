import cv2
import numpy as np
import os
from datetime import datetime

from PIL import Image


class EnvRenderer:
    def __init__(self, env, delay=50):
        self.env = env
        self.images = []
        self.delay = delay

    def record(self, img=None):
        img = self.generate_img() if img is None else img
        self.images.append(img)

    def generate_img(self):
        self.image = Image.fromarray(self.env.matrix, 'RGB')
        self.image = self.image.resize((500, 500), Image.NONE)
        return self.image

    def render(self, record=True):
        img = self.generate_img()

        cv2.namedWindow('SnakeAI', cv2.WINDOW_NORMAL)
        cv2.imshow('SnakeAI', np.array(img))
        cv2.waitKey(delay=self.delay)

        if record:
            self.images.append(img)

    def destroy_window(self):
        cv2.destroyWindow('SnakeAI')

    def save_images(self, path='./game_images'):
        if len(self.images) == 0:
            return
        if not os.path.exists(path):
            os.mkdir(path)
        path = path + '/' + str(datetime.now()).replace(' ', '').replace('.', '').replace('-', '').replace(':', '')
        if not os.path.exists(path):
            os.mkdir(path)
        for i, img in enumerate(self.images):
            cv2.imwrite(f"{str(path)}/{str(i)}.png", np.array(img))

    def save_video(self, path='./game_videos'):
        if len(self.images) == 0:
            return
        if not os.path.exists(path):
            os.mkdir(path)
        filename = str(datetime.now()).replace(' ', '').replace('.', '').replace('-', '').replace(':', '')
        out = cv2.VideoWriter(f'{path}/v{filename}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (500, 500))
        for img in self.images:
            out.write(np.array(img))
        out.release()
