NAME=aisl-pytorch
VERSION=20.03-py3
CONTAINER_NAME=aisl-pytorch

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
    		--gpus="device=2" \
    		-v ${PWD}/training:/root/training/ \
    		-v /data/aisl/matsuzaki/dataset:/tmp/dataset \
    		-v /data/aisl/matsuzaki/runs/:/tmp/runs/ \
		--rm \
		--shm-size 1G \
		--workdir /root/training/ \
		--name $(CONTAINER_NAME) \
		$(NAME)-train:$(VERSION) \
    python test.py