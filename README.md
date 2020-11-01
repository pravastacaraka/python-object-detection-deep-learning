# Object Detection Deep Learning Using NVIDIA GPU
To learn how to use OpenCV’s ```dnn``` module with an NVIDIA GPU for up to 1,549% faster object detection (YOLO and SSDs) and instance segmentation (Mask R-CNN). We were able to push a given network’s computation from the CPU to the GPU.

Inside this tutorial you’ll learn how to implement Single Shot Detectors (SSDs), YOLO, and Mask R-CNN using OpenCV’s “deep neural network” (dnn) module and an NVIDIA/CUDA-enabled GPU.

## Absolute Pre-requisites
* An NVIDIA GPU
* [CUDA Toolkit v10.0](https://developer.nvidia.com/cuda-10.0-download-archive) installed.
* [cuDNN v7.6.4 for CUDA v10.0](https://developer.nvidia.com/rdp/cudnn-archive) configured and installed.
* [OpenCV](https://github.com/opencv/opencv/archive/4.2.0.zip) and [OpenCV Contrib](https://github.com/opencv/opencv_contrib/archive/4.2.0.zip) v4.2.0 or above, we built OpenCV from scratch because the ```pip```'s OpenCV doesn't support GPU.

If you haven't installed one or all of them, please follow the tutorial from this [link](https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/).

## Project Structure
Before we review the structure of today’s project, grab the project first.

```bash
$ git clone https://github.com/pravastacaraka/python-object-detection-deep-learning.git
```

From there, use the ```tree``` command in your terminal to inspect the project hierarchy:

``` bash
$ tree --dirsfirst
.
├── images
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   ├── 5.jpg
│   ├── 6.jpg
│   ├── car.mp4
│   ├── cat.mp4
│   └── soccer.jpeg
├── opencv-ssd
│   ├── main_image.py
│   ├── main_live.py
│   └── model
│       ├── MobileNetSSD_deploy.caffemodel
│       └── MobileNetSSD_deploy.prototxt.txt
├── opencv-yolo
│   ├── main_image.py
│   ├── main_live.py
│   └── model
│       ├── coco.names
│       ├── yolov3.cfg
│       └── yolov3.weights
└── output

6 directories, 18 files
```

In today’s tutorial, we will review three Python script folders:

* ```opencv-ssd```: Performs Caffe-based MobileNet SSD object detection on 20 COCO classes with CUDA.
* ```opencv-yolo```: Performs YOLO V3 object detection on 80 COCO classes with CUDA.
* ```opencv-mask-r-cnn```: Performs TensorFlow-based Inception V2 segmentation on 90 COCO classes with CUDA.

Each of the model files and class name files are included in their respective folders with the exception of our MobileNet SSD (the class names are hardcoded in a Python list directly in the script).

## Implementing Single Shot Detectors (SSDs)
Execute the following command to obtain a baseline for our SSD by running it on our CPU:

```bash
$ python main_live.py \
  --prototxt model/MobileNetSSD_deploy.prototxt.txt \
  --model model/MobileNetSSD_deploy.caffemodel \
  --input ../images/car.mp4 \
  --output ../images/car.mp4 \
  --display 0
```

To see the detector really fly, let’s supply the ```--use-gpu 1``` command line argument, instructing OpenCV to push the ```dnn``` computation with our NVIDIA GPU:

```bash
$ python main_live.py \
  --prototxt model/MobileNetSSD_deploy.prototxt.txt \
  --model model/MobileNetSSD_deploy.caffemodel \
  --input ../images/car.mp4 \
  --output ../images/car.mp4 \
  --display 0 \
  --use-gpu 1
```
If you want to detect from image, execute the ```main_image.py``` script.

## Implementing YOLOv3
Execute the following command to obtain a baseline for our YOLOv3 by running it on our CPU:

```bash
$ python main_live.py \
  --yolo model \
  --input ../images/car.mp4 \
  --output ../images/car.mp4 \
  --display 0
```

To see the detector really fly, let’s supply the ```--use-gpu 1``` command line argument, instructing OpenCV to push the ```dnn``` computation with our NVIDIA GPU:

```bash
$ python main_live.py \
  --yolo model \
  --input ../images/car.mp4 \
  --output ../images/car.mp4 \
  --display 0
  --use-gpu 1
```
If you want to detect from image, execute the ```main_image.py``` script.

## Implementing Mask R-CNN
Coming soon...
