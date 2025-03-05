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
To deal with haze in images, All-in-One Dehazing Network (AOD-Net) [3] can be used as a solution. AOD-Net represents a paradigm shift in dehazing by formulating the problem as an end-to-end trainable task. Unlike previous methods, AOD-Net does not estimate intermediate parameters like the transmission matrix or atmospheric light separately. Instead, it directly predicts the clean image from the hazy image using a lightweight CNN. This approach simplifies the dehazing process and makes it easier to integrate AOD-Net into larger pipelines,such as YOLO.  
There are several implementation of AOD-Net algorithm, I choose to use pytorch version of [AOD-Net](https://github.com/kivenyangming/AOD-Net_pytorch).  
In my program, I create a AODNet block, which can be embedded into backbone of YOLO.
``` python
class AODNet(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(c1, 3, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        self.output_conv = nn.Conv2d(3, c2, 1, 1, 0, bias=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)

        return self.output_conv(clean_image)
```
And it is added at the top of backbone, process the input directly.  
``` yaml
  - [-1, 1, AODNet, [3]] # Add Dehaze Block
```

#### CBAM (Convolutional Block Attention Module)
CBAM (Convolutional Block Attention Module) is a lightweight and effective attention mechanism designed to improve the feature extraction capability of convolutional neural networks (CNNs).  
CBAM enhances the network's performance by sequentially applying **channel attention** and **spatial attention**, allowing the model to focus on the most important features in both dimensions.  
To improve the performance of AOD-Net, I add some CBAM blocks in the head of YOLO net.  
CBAM block is provided by [ultralytics](https://github.com/ultralytics/ultralytics) library, and can be used directly.  
``` python
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
```
I embed CBAM block in YOLO head with this format.  
``` yaml
  - [ -1, 1, CBAM, [3]] # add CBAM in P4
```

## Results
### Speed Estimation
The Perspective Transformation method is not accurate enough to calculate the real speed. But it is enough to alert when speed of vehicle is higher than a small threshold. In my sample program, the `speed_threshold` is set to `2 m/s`.  
There are 2 main cases in the speed estimation part. I provide screenshot in an operated video to show both of cases.  
1. **Speed over threshold**  
When speed is over threshold, the system will alert. In practical applications, the system will alert through a loudspeaker or buzzer, which can inform the blinds about the danger. Here, a red alert marker is used to indicate the warning, which is more intuitive in system demonstration.
![Speed over threshold](/Speed_over_threshold.png "Speed over threshold")  

2. **Speed below threshold**  
When speed is below threshold, all the vehicles are very slow, which means it will be safe to go across the roads. No alert will occur.  
![Speed below threshold](/Speed_below_threshold.png "Speed below threshold")  

### Object detection in haze
I use AOD-Net and CBAM to modify the structure of YOLO. Several combinations of AOD-Net and CBAM have been tried, the following one turned out to have the best effect.  
AOD-Net is put on the start of backbone, following the input. The input and output of AOD-Net are both 3-channel, which guarantee further structure of YOLO will not be changed.  
``` yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, AODNet, [3]] # Add Dehaze Block
```
The best version of model use 2 CBAM blocks, added in P3 and P4 separately. Here I choose to set `kernel_size` to 3, because in the project, we focus more on small targets (such as vehicles and pedestrians). It is faster and more suitable for small targets. 
``` yaml
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [ -1, 1, CBAM, [3]] # add CBAM in P4
  - [-1, 3, C2f, [512]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [ -1, 1, CBAM, [3]]  # add CBAM in P3
  - [-1, 3, C2f, [256]] # 16 (P3/8-small)
```
The final version for dehaze use `yolov10n-dehaze-add-head1.yaml` as configuration, and the output is stored in `results/log_train21_add2CBAM.out`.  
``` out
Ultralytics YOLOv8.1.34 🚀 Python-3.9.20 torch-2.0.1+cu117 CUDA:0 (Tesla P100-SXM2-16GB, 16276MiB)
YOLOv10n-dehaze-add-head1 summary (fused): 309 layers, 2871395 parameters, 0 gradients, 14.0 GFLOPs
                   all        100        773      0.603      0.398      0.449      0.252
            pedestrian        100        173      0.605      0.301      0.376      0.159
                   car        100        479      0.708      0.674      0.722      0.432
                   bus        100         49      0.625      0.531      0.561      0.332
               bicycle        100         12      0.471      0.167      0.211      0.136
             motorbike        100         60      0.606      0.317      0.375      0.199
Speed: 1.4ms preprocess, 6.7ms inference, 0.0ms loss, 0.0ms postprocess per image
```
The base model use default YOLO configuration, which is `yolov10n-test.yaml`, stored in `results/log_train19_base.out`.
``` out
Ultralytics YOLOv8.1.34 🚀 Python-3.9.20 torch-2.0.1+cu117 CUDA:0 (Tesla P100-SXM2-16GB, 16276MiB)
YOLOv10n-test summary (fused): 285 layers, 2696366 parameters, 0 gradients, 8.2 GFLOPs
                   all        100        773      0.526      0.417      0.452      0.249
            pedestrian        100        173      0.559      0.417      0.393      0.163
                   car        100        479      0.617      0.689        0.7      0.422
                   bus        100         49      0.597      0.531      0.584      0.362
               bicycle        100         12      0.408     0.0833      0.222       0.11
             motorbike        100         60      0.448      0.367       0.36      0.189
Speed: 1.5ms preprocess, 2.5ms inference, 0.0ms loss, 0.0ms postprocess per image
```

#### **Overall Performance Comparison**
| Metric              | modified model | base model | Change      |
|---------------------|----------------|------------|-------------|
| **mAP@0.5 (all)**   | **0.603**      | 0.526      | **↑ 0.077** |
| **Precision (all)** | 0.398          | **0.417**  | **↓ 0.019** |
| **Recall (all)**    | 0.449          | **0.452**  | **↓ 0.003** |
| **F1-score (all)**  | **0.252**      | 0.249      | **↑ 0.003** |

**mAP@0.5 increased significantly by 7.7%, indicating improved overall detection accuracy.**  
Precision, Recall, and F1-score remained similar, the tiny difference is probably caused by the quality of dataset.

---

#### **Category-wise Comparison**
| Category       | mAP@0.5 (New) | mAP@0.5 (Old) | Change      | Recall Change |
|----------------|---------------|---------------|-------------|---------------|
| **Pedestrian** | 0.605         | 0.559         | **↑ 0.046** | **↓ 0.116**   |
| **Car**        | **0.708**     | 0.617         | **↑ 0.091** | **↓ 0.015**   |
| **Bus**        | **0.625**     | 0.597         | **↑ 0.028** | **0**         |
| **Bicycle**    | **0.471**     | 0.408         | **↑ 0.063** | **↑ 0.084**   |
| **Motorbike**  | **0.606**     | 0.448         | **↑ 0.158** | **↓ 0.05**    |

**Significant mAP improvement for cars, buses, bicycles and motorbikes, especially a 9.1% boost for cars.**  
The performance of pedestrian detection dropped. Fortunately, the system is used for vehicle detection, so the accuracy of pedestrian detection can be ignored.  

---
