import cv2
import os

# root_dir = 'C://Users/cathx/repos/argoverse-api/tracking_train1_v1.1.tar/argoverse-tracking/train1/3d20ae25-5b29-320d-8bae-f03e9dc177b9/'
root_dir = 'C://Users/cathx/repos/argoverse-api/tracking_train1_v1.1.tar/argoverse-tracking/train1/'
#
# image_folder = 'C://Users/cathx/repos/argoverse-api/tracking_train1_v1.1.tar/argoverse-tracking/train1/3d20ae25-5b29-320d-8bae-f03e9dc177b9/stereo_front_right/'
# videos = 'videos/'
# video_name = 'test_vid_r1.avi'
# vsObj = cv2.VideoWriter(os.path.join(root_dir+videos+video_name))
# path = os.path.join(root_dir, videos)
# # os.mkdir(path)
#
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
#
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# fps = 5
# video = cv2.VideoWriter(os.path.join(path+ video_name), 0, fps, (width, height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#

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
