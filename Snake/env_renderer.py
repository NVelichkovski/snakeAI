import cv2
import numpy as np
import os
from datetime import datetime


class CV2Renderer:
    """
    Renderer for the SnakeMaze

    It can generate, render and save frames from the current state of the maze into the memory. later the frames could
    be stored on disk as images or video.
    """

    def __init__(self, env, delay=50, window_size=600, image_size=(500, 500)):
        """
        Constructor for EnvRenderer object

        :param env: Environment
            The instance of the current Snake
        :param delay: int, Optional
            The delay between two frames
        """
        self.env = env
        self.images = []
        self.delay = delay
        self.window_size = window_size
        self.image_size = image_size

    def record(self, img=None):
        """
        Store image in memory

        :param img: Image, Optional
            The image that should be stored in self.images
        :return: None
        """
        img = self.generate_np_img() if img is None else img
        self.images.append(img)

    def generate_np_img(self, handle=0):
        """
        Generate image from the current state of the maze

        :return: Image
            The image generated from the maze
        """
        self.image = cv2.resize(self.env.snake_matrices[handle], self.image_size, interpolation=cv2.INTER_AREA)
        return self.image

    def render(self, record=True):
        """
        Render and show the image generated from the current state of the maze

        If record is true the image is stored in memory

        :param record: boolean, Optional
            If record is True the image is stored in memory
        :return: None
        """
        img = self.generate_np_img()

        cv2.namedWindow('snakeAI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('snakeAI', self.window_size, self.window_size)
        cv2.imshow('snakeAI', img)
        cv2.waitKey(delay=self.delay)

        if record:
            self.images.append(img)

    def destroy_window(self, window_name='snakeAI'):
        cv2.destroyWindow(window_name)

    def save_images(self, path=None):
        """
        Save the images stored in self.images

        The images are saved in a folder named by the current timestamp. The path to the folder is defined by path
        :param path: String, Optional
            Path to the folder where the images will be stored. The default path is ./game_images
        :return: None
        """
        if len(self.images) == 0:
            return
        if path is None:
            path = os.path.join('game_images', datetime.now().strftime('%d%h%Y__%H%M%S%f'))

        os.makedirs(path, exist_ok=True)

        for i, img in enumerate(self.images):
            cv2.imwrite(os.path.join(path, f'{i}.png'), np.array(img))

    def save_video(self, filename=None):
        """
        Save video from the images stored in self.images

        The video is saved in a folder named by the current timestamp. The path to the folder is defined by path
        :param path: String, Optional
            Path to the folder where the video will be stored. The default path is ./game_videos
        :return: None
        """
        if len(self.images) == 0:
            return

        if filename is None:
            filename = os.path.join('game_videos', datetime.now().strftime('%d%h%Y__%H%M%S%f') + '.mp4')

        if not filename.endswith('.mp4'):
            filename += '.mp4'

        os.makedirs(os.path.split(filename)[0], exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, self.image_size, )

        for img in self.images:
            out.write(img)
        out.release()

    def show_cache(self):
        for img in self.images:
            print(img.shape)
            cv2.namedWindow('snakeAI Cache', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('snakeAI Cache', self.window_size, self.window_size)
            cv2.imshow('snakeAI Cache', img)
            cv2.waitKey(delay=self.delay)
        self.destroy_window('snakeAI Cache')
