# 4D Spatio-Temporal Object Detection
### A comparison between multiple 3D object detection models and which one is more robust for 4D spatio-temporal object detection

3D object detection has been deemed a crucial task in autonomous driving. So far, LiDAR has been the go-to technique for collecting 3D point clouds and using 3D object detection models for inference on them. Recently, however, we have seen an increase in research on [PseudoLiDAR](https://arxiv.org/pdf/1812.07179.pdf) techniques that focus on creating point clouds using input from camera streams. Here is how the pipeline is:

1. The point clouds from the dataset are used to generate disparity using stereo images.
2. The model is trained on those disparities so that at test time, we can get stereo images and predict the disparity between pixels in left and right images.
3. The disparities are then converted into, PseudoLiDAR point clouds for 3D representation using the techniques described in the [PseudoLiDAR](https://arxiv.org/pdf/1812.07179.pdf) paper.
4. Any 3D object detection model can be used to make predictions on the point clouds.

Two main techniques that I have been working with are Frustum PointNets and SparseNets. [PointNet](https://arxiv.org/pdf/1612.00593.pdf) was developed to process point clouds for 3D classification and segmentation where it takes in points from the point cloud and outputs either class labels for the entire input or per point segment/part labels for each point of the input(segmentation). The network focuses on max pooling where it learns functions/criteria that select interesting or informative points in the point cloud and encode the reason for their selection.

One of the shortcomings of the PointNet architecture is that while they are able to perform point-level segmentation, they are unable to perform instance segmentation for 3D object detection. This problem is solved by Furstum PointNets. They perform object detection on 2D images but then, that bounding box is lifted to 3D representation in point clouds. These 3D frustums have 3D space trimmed and perform 3D object instance segmentation and amodal 3D bounding box regression using PointNet. This not only lowers the 3D space that you have to look for in object, but it also helps with data representation of 3D models.

Another 3D object detection model that works along the same lines is SparseNet which makes use of the Minkowski Engine to create sparse tensors from 3D data. It works on the intuition that 3D space is huge and there is a lot of noise in the data. So, we can create sparse tensor from 3D point clouds where most of the points are 0. Then, we train a SparseNet on the Sparse tensors for object detection.

The goal with this project is to compare the two 3D object detection techniques and see which one is more applicable for real-time object detection on PseudoLiDAR point clouds. 
