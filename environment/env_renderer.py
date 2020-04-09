import cv2
import numpy as np
import os
from datetime import datetime

from PIL import Image


class CV2Renderer:
    """
    Renderer for the SnakeMaze

    It can generate, render and save frames from the current state of the maze into the memory. later the frames could
    be stored on disk as images or video.
    """

    def __init__(self, env, delay=50):
        """
        Constructor for EnvRenderer object

        :param env: Environment
            The instance of the current environment
        :param delay: int, Optional
            The delay between two frames
        """
        self.env = env
        self.images = []
        self.delay = delay

    def record(self, img=None):
        """
        Store image in memory

        :param img: Image, Optional
            The image that should be stored in self.images
        :return: None
        """
        img = self.generate_img() if img is None else img
        self.images.append(img)

    def generate_img(self):
        """
        Generate image from the current state of the maze

        :return: Image
            The image generated from the maze
        """
        self.image = Image.fromarray(self.env.snake_matrices[1], 'RGB')
        self.image = self.image.resize((500, 500), Image.NONE)
        return self.image

    def render(self, record=True):
        """
        Render and show the image generated from the current state of the maze

        If record is true the image is stored in memory

        :param record: boolean, Optional
            If record is True the image is stored in memory
        :return: None
        """
        img = self.generate_img()

        cv2.namedWindow('SnakeAI', cv2.WINDOW_NORMAL)
        cv2.imshow('SnakeAI', np.array(img))
        cv2.waitKey(delay=self.delay)

        if record:
            self.images.append(img)

    def destroy_window(self):
        cv2.destroyWindow('SnakeAI')

    def save_images(self, path='./game_images'):
        """
        Save the images stored in self.images

        The images are saved in a folder named by the current timestamp. The path to the folder is defined by path
        :param path: String, Optional
            Path to the folder where the images will be stored. The default path is ./game_images
        :return: None
        """
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
        """
        Save video from the images stored in self.images

        The video is saved in a folder named by the current timestamp. The path to the folder is defined by path
        :param path: String, Optional
            Path to the folder where the video will be stored. The default path is ./game_videos
        :return: None
        """
        if len(self.images) == 0:
            return
        if not os.path.exists(path):
            os.mkdir(path)
        filename = str(datetime.now()).replace(' ', '').replace('.', '').replace('-', '').replace(':', '')
        out = cv2.VideoWriter(f'{path}/v{filename}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (500, 500))
        for img in self.images:
            out.write(np.array(img))
        out.release()
