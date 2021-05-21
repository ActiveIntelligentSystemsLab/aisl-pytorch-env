#!/bin/bash
set -e

g_setup=/opt/ros/$ROS_DISTRO/setup.bash
l_setup=/root/catkin_ws/devel/setup.bash

if [ -f "$g_setup" ]; then
  source $g_setup
else
  echo "$g_setup not found"
fi

if [ -f "$l_setup" ]; then
  source $l_setup
else
  echo "$l_setup not found"
fi

network_if=eth0
export TARGET_IP=$(LANG=C /sbin/ifconfig $network_if | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*')
if [ -z "$TARGET_IP" ] ; then
      echo "ROS_IP is not set."
    else
          export ROS_IP=$TARGET_IP
fi

exec "$@"