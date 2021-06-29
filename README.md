# YOLOV5-DeepSORT-Vehicle-Tracking-Master
In this project, urban traffic videos are collected from the middle section of Xi 'an South Second Ring Road with a large traffic flow, and interval frames are extracted from the videos to produce data sets for training and verification of YOLO V5 neural network. Combined with the detection results, the open-source vehicle depth model data set is used to train the vehicle depth feature weight file, and the deep-sort algorithm is used to complete the target tracking, which can realize real-time and relatively accurate multi-target recognition and tracking of moving vehicles.
And I also upload my dissertation of bachelor degree, I think you can learn about how to set up this project via this paper (But it is in simplified Chinese.).

# Installation
Python 3.6 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:
```bash
pip install -r requirements.txt
```

# Run Tracker
After you download this project, please download the weight of YOLO V5 model and Deep-SORT model respectively. 
You can download the YOLO weight trained by me from https://drive.google.com/file/d/1-8Xm3eUMMJF5XNiF649kqnqoYeWhv3kT/view?usp=sharing or choose to download the pretrained weight of the YOLO V5 model with using the `./yolov5/weights/downloadweight.sh`.

You can download the Deep-SORT weight trained by me from https://drive.google.com/file/d/1-GjB1pGudjM70C1Jk7WhWUWShzCvvvra/view?usp=sharing or choose to download the pretrained weight of the Deep-SORT model from https://drive.google.com/file/d/1-EsIHxEqr9elRryUqPBcBBAY3qoIEbZS/view?usp=sharing . And then put the weight into the folder namely `/deep_sort_pytorch/deep_sort/deep/checkpoint`.

Finanlly, you can choose to download the test video producted by me from https://drive.google.com/file/d/1cMch88P8xZ95SoHN88aC53oiOLlD3Egb/view?usp=sharing.

Tracking can be run on most video formats

```bash
python3 track.py --source ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

MOT compliant results can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```

# Result
https://drive.google.com/file/d/1-2hL9UE3i24bRE0an9HTDbN5bJBEeXfg/view?usp=sharing
![1](https://user-images.githubusercontent.com/64308326/119247613-9fd9db80-bbbd-11eb-9280-4a687e2a8250.jpg)

# Train YOLO weights with your datasets
Please have a reference on https://github.com/ultralytics/yolov5.

# Train Deep-SORT weights with your datasets
I have update my datasets about the appearance of cars from https://drive.google.com/file/d/1lushuv4QMTmfFwURU1Ug6mezkzQAMi0S/view?usp=sharing. You can choose to read the original paper of Deep-SORT to learn how to train it. I will also update quickly.
Updating now......
I consider to create a new repository about it now......

# References
https://github.com/ultralytics/yolov5 
Simple Online and Realtime Tracking with a Deep Association Metric https://arxiv.org/abs/1703.07402
