# aisl-pytorch-env
Docker + ROS environment for easily using existing pre-trained PyTorch models

## Requirements
- Docker
- nvidia-docker2
- NVIDIA driver compatible with 

## Building docker images

### Training

``` make build-training ```

### ROS

``` make build-ros ```

## Training

1. Put a dataset somehwhere.
2. Edit `DATASET_DIR` and `LOG_DIR` in `Makefile`.
   - `DATASET_DIR` : Location of the dataset
   - `LOG_DIR` : Location to put training log (tensorboard) and checkpoints (trained weights)
3. Run `make train`
 
## Running ROS nodes

### Common things

1. Build ROS packages
   ```
   make catkin-build
   ```
2. Run ROS master
   ```
   docker-compose up master
   ```
3. Run `usb_cam`
   ```
   docker-compose up ros-usb-cam
   ```
### Object recognition
- Run:
   ```
   docker-compose up ros-object-recognition
   ```
### Object detection
- Run:
   ```
   docker-compose up ros-object-detection
   ```

## Models to be supported

- [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)
- [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/)
- [DEEPLABV3-RESNET101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)
- [FCN-RESNET101](https://pytorch.org/hub/pytorch_vision_fcn_resnet101/)