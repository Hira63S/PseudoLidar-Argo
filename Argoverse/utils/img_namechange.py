""" Changing image file names to KITTI format"""
import os

def image_names(root_dir):

  for i in os.listdir(root_dir):
      path = (root_dir + i + '/stereo_front_right/')

      for count, filename in enumerate(os.listdir(root_dir + i + '/stereo_front_right')):
          new_name = "{:06}.jpg".format((count))
          source = path + filename
          new_name = path + new_name

          os.rename(source, new_name)


if __name__ == '__main__':
    main()
