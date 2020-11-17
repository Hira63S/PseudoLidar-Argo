"""Converting images to videos for 4D detection """

import os
import cv2


def videos(root_dir):
    for i in os.listdir(root_dir):
        path = os.path.join(root_dir + i + '/stereo_front_left/')
        # print(os.listdir(path))
        images = [img for img in os.listdir(path) if img.endswith(".jpg")]

        frame = cv2.imread(os.path.join(path, images[0]))
        height, width, layers = frame.shape

        # video_name = 'left_cam.avi'
        fps = 5
        video = cv2.VideoWriter(os.path.join(root_dir+i+'/left_cam.avi'), 0, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(path, image)))

    cv2.destroyAllWindows()
    video.release()

videos(root_dir)
