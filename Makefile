NAME=aisl-pytorch
CONTAINER_NAME=aisl-pytorch
#VERSION=21.03-py3
VERSION=20.03-py3
ROS_DISTRO=noetic
DATASET_DIR=/data/aisl/matsuzaki/dataset
LOG_DIR=/data/aisl/matsuzaki/runs/

build-train: 
	docker build -t $(NAME)-train:$(VERSION) -f ./docker/Dockerfile_base \
		--build-arg VERSION=$(VERSION) \
		.

build-ros : 
	docker build -t $(NAME)-ros:$(VERSION) -f ./docker/Dockerfile_ros \
		--build-arg VERSION=$(VERSION) \
		.

restart: stop start

start:
	docker start $(CONTAINER_NAME)

contener=`docker ps -a -q`
image=`docker images | awk '/^<none>/ { print $$3 }'`
	
clean:
	@if [ "$(image)" != "" ] ; then \
		docker rmi $(image); \
	fi
	@if [ "$(contener)" != "" ] ; then \
		docker rm $(contener); \
	fi
	
stop:
	docker stop $(CONTAINER_NAME)
	
rm:
	docker rm -f $(CONTAINER_NAME)
attach:
	docker start $(CONTAINER_NAME) && docker exec -it $(CONTAINER_NAME) /bin/bash
	
logs:
	docker logs $(CONTAINER_NAME)

#
# Execusion commands
#
efficientnet-test:
	docker run -it \
		--gpus="device=0" \
		-v ${PWD}/training:/root/training/ \
		--rm \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME) \
		$(NAME)-train:$(VERSION) \
		python test.py

train:
	docker run -it \
		--gpus="device=2" \
		-v ${PWD}/training:/root/training/ \
		-v ${DATASET_DIR}:/tmp/dataset \
		-v ${LOG_DIR}:/tmp/runs/ \
		--rm \
		--shm-size 1G \
		--workdir /root/training/ \
		--name $(NAME)-train \
		$(NAME)-train:$(VERSION) \
		python train.py

master:
	docker run -it \
		--rm \
		--workdir /root/training/ \
		-e ROS_MASTER_URI=http://localhost:11311 \
		--name aisl-pytorch-ros-master \
		$(NAME)-ros:$(VERSION) \
		roscore

catkin-build:
	docker run -it \
		-v ${PWD}/training:/root/training/ \
		-v ${PWD}/catkin_ws:/root/catkin_ws/ \
		--rm \
		--workdir /root/catkin_ws/ \
		--name catkin-build \
		$(NAME)-ros:$(VERSION) \
		catkin build -DCMAKE_BUILD_TYPE=Debug

ros-object-recognition:
	docker run -it \
		--gpus="device=0" \
		-v ${PWD}/training:/root/training/ \
		-v ${PWD}/catkin_ws:/root/catkin_ws/ \
		--rm \
		--shm-size 1G \
		--workdir /root/catkin_ws/ \
		-e ROS_MASTER_URI=http://aisl-pytorch-ros-master:11311 \
		--name ros-object-recognition \
		$(NAME)-ros:$(VERSION) \
		roslaunch pytorch_ros object_recognition.launch
