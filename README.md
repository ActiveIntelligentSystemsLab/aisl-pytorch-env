# aisl-pytorch-env
Docker + ROS environment for easily using existing pre-trained PyTorch models

<img width=400 src="images/demo.gif" alt="Demo"></img>

## Environment
- CUDA 11.2
- PyTorch 1.8.1
- Torchvision 0.9.1

## Requirements
- Docker
- nvidia-docker2
- NVIDIA driver compatible with CUDA 11.2

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

 ### Semantic segmentation
- Run:
   ```
   docker-compose up ros-semantic-segmentation
  ```

 ### Depth estimation ([MiDaS](https://pytorch.org/hub/intelisl_midas_v2/))
- Run:
   ```
   docker-compose up ros-depth-estimation
   ```

## More things to add

- Face detection / recognition