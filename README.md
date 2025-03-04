# A Blind Aiding System based on YOLO

## Aims for the project
Nowadays, the roads are busier than ever before, for the increasing of all kinds of vehicles like motors, cars, etc. Even for healthy people, going cross the roads can be hard in a degree. From a very young age, I was told that observing road conditions is necessary to be done before going across the road.  
For blind people, it is very hard to confirm the safety when going cross the road. There will be plenty of cars on the road, the information about these cars should be informed to them in an efficient and reliable way.
![the blind on road](/the_blind_on_road.png "the blind on road")

## Custom YOLO Version
This repository is a fork of [YOLOv10](https://github.com/THU-MIG/yolov10) and has been modified to suit specific needs. The project remains under the **AGPL-3.0 License**, ensuring that any modifications remain open-source.


## Experimental environment
### Dataset
I upload the dataset to huggingface. The dataset contains 5 labels, which are pedestrian, car, bus, bicycle and motorbike.  
Dataset: [Vehicle-Detection-in-haze-Dataset](https://huggingface.co/datasets/wandering-tiger/Vehicle-Detection-in-haze-Dataset)

### Hardware specifications
I use a cluster provided by Heriot-Watt University in the cloud comprising 1 GPU and 24 CPUs.  

- CPU: 24x Intel Xeon Platinum 8167M. Base frequency 2.0 GHz, max turbo frequency 2.4 GHz.
- CPU Memory: 72GB
- GPU: 1x NVIDIA Tesla P100
- GPU Memory: 16GB

## Main works on creating the system
### Object Detection
I choose to the use YOLOv10 for object detection, which is used for detecting vehicles.  
The system is built based on YOLO. It should detect vehicles automatically, and alert when speed of vehicles over a threshold.  
Firstly, I forked YOLOv10 and deploy on server, trained a model with my own dataset, which can detect objects successfully. Further work focus on the improvement of the detection and model.

### Object Tracking
After detecting objects, object tracking should be applied before speed estimation.  
Here I choose to use `BYTETrack` in [Supervision](https://github.com/roboflow/supervision?ref=blog.roboflow.com) library for object tracking.

### Speed Estimation
Speed estimation is essential in the project. Generally, there are two main ways for speed estimation.  
1. **Common Video-Based Vehicle Speed Measurement Method**  
One standard approach calculates the actual physical displacement of a vehicle between two frames based on parameters such as the camera's vertical height from the ground and focal length. The displacement is then divided by the time between frames to determine the vehicle's speed.
However, this method requires precise knowledge of the camera’s attributes at each measurement location, which is often difficult to obtain in real-world scenarios.
2. **Perspective Transformation-Based Vehicle Speed Measurement**  
To overcome the challenges of parameter dependency, an alternative method uses perspective transformation to convert the video frames into a bird’s-eye view, where each pixel represents a fixed physical distance. This ensures that any object’s motion can be measured directly using pixel displacement and time difference.

Obviously, in our scenario, Method 1 is not suitable, because we are not able to obtain specified location information about cameras and cars.
**Perspective Transformation** is needed for estimation. We can simulate actual situations by taking photos, manually calculate the transformation matrix, and apply it to the program.  
For Perspective Transformation, a [blog](https://blog.roboflow.com/estimate-speed-computer-vision/) provides an excellent method, and can adapt to our own scenarios. I refer to the methods in the blog, which get satisfying performance.  

#### Perspective Transformation
![picture of Perspective Transformation](/Perspective_Transformation_sample.jpg "Perspective Transformation Sample")
We need a way to transform the coordinates in the image, which is represented by pixels, into actual coordinates on the road, removing the perspective-related distortion along the way. Fortunately, we can do this with OpenCV and some mathematics.  
To transform the perspective, we need a Transformation Matrix, which we determine using the `getPerspectiveTransform` function in OpenCV. This function takes two arguments - source and target regions of interest. In the visualization below, these regions are labeled `A-B-C-D` and `A'-B'-C'-D'`, respectively.  
In this example, I reorganize the coordinates of vertices `A-B-C-D` and `A'-B'-C'-D'` into 2D `SOURCE` and `TARGET` matrices, respectively, where each row of the matrix contains the coordinates of one point.  
``` python
SOURCE = np.array([
    [662*ratio_width, 753*ratio_height],
    [1109*ratio_width, 737*ratio_height],
    [1588*ratio_width, 934*ratio_height],
    [362*ratio_width, 1010*ratio_height]
])

TARGET = np.array([
    [0, 0],
    [5.3, 0],
    [5.3, 9.7],
    [0, 9.7],
])
```
The `ratio_width` and `ratio_height` coefficient multiplied above, are used for resize real picture to the size of `SOURCE`, which is necessary for adapt the Perspective Transformation to different size.

#### Speed Calculation
We could calculate our speed every frame: calculate the distance traveled between two video frames and divide it by the inverse of our FPS, in my case, 1/25. Unfortunately, this method can result in very unstable and unrealistic speed values.  
To prevent this, we average the values obtained throughout one second. This way, the distance covered by the car is significantly larger than the small box movement caused by flickering, and our speed measurements are closer to the truth.  
``` python
for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
    class_name = CLASS_NAMES.get(class_id, "Unknown")
    if len(coordinates[tracker_id]) < video_info.fps / 2:
        labels.append(f"#{tracker_id} ({class_name})")
    else:
        # calculate speed
        coordinate_start = coordinates[tracker_id][-1]
        coordinate_end = coordinates[tracker_id][0]
        distance = abs(coordinate_start - coordinate_end)
        time = len(coordinates[tracker_id]) / video_info.fps
        speed = distance / time
        if speed > speed_threshold:
            labels.append(f"#{tracker_id} ({class_name}) {int(speed)} m/s ALERT!")
            alert_detections.append(tracker_id)
        else:
            labels.append(f"#{tracker_id} ({class_name}) {int(speed)} m/s")
```

### Special conditions handling
After estimating speed successfully, we can focus on improving the model, which can increase the accuracy of detection.  
As a blind aiding system, some special conditions should be taken into consider. YOLO has a basic structure, which perform well in most situations, but we can change the layers of net in both `backbone` and `head`, to get better performance in conditions like haze.  
My dataset contains the haze environment, which will influence the performance of YOLO, so a dehaze net is needed. AOD-Net is well known dehaze net, which can be combined with YOLO. It is designed based on a re-formulated atmospheric scattering model, partly preserve the physical model, and can improve the performance of model.  

#### AOD-Net


#### CBAM (Convolutional Block Attention Module)

## Results

