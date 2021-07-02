#!/bin/sh
# docker_ros_setup.sh
#   set the ROS-related environment variables
#   according to the docker container running 'roscore'

#
# Get an (virtual) IP address of a docker container named "master"
#

if [ "$1" != "" ]; then
  CONTAINER_NAME=$1
else
  CONTAINER_NAME="master-aisl-pytorch"
fi

IP_ADDRESS=`docker inspect $CONTAINER_NAME | grep -E "IPAddress" | grep -m1 -o "[0-9]\+.[0-9]\+.[0-9]\+.[0-9]\+"`

if [ "$IP_ADDRESS" != "" ]; then
  export ROS_MASTER_URI=http://$IP_ADDRESS:11311
  export ROS_IP=${IP_ADDRESS%.*}.1

  echo "Docker ROS setup : $IP_ADDRESS"
  echo "ROS IP           : $ROS_IP"
else
  echo CONTAINER \"$CONTAINER_NAME\" not found
fi

