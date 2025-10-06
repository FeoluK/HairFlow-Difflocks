#!/usr/bin/env bash

# Check args
if [ "$#" -ne 0 ]; then
	echo "usage: ./run.sh"
	exit 1
fi

# Get this script's path
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

set -e

# --volume $SSH_AUTH_SOCK:/ssh-agent \
# --env SSH_AUTH_SOCK=/ssh-agent \


# for more info see: https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
# for more info see: https://gist.github.com/RafaelPalomar/f594933bb5c07184408c480184c2afb4
# Run the container with shared X11
docker run\
	--rm \
	--shm-size 12G\
	--gpus all\
	--net host\
	--privileged\
	-e SHELL\
	-e DISPLAY\
	-e DOCKER=1\
	-v /dev:/dev\
	--volume=/run/user/${USER_UID}/pulse:/run/user/1000/pulse \
	-e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
	-v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native \
	-v ~/.config/pulse/cookie:/root/.config/pulse/cookie \
	--group-add $(getent group audio | cut -d: -f3) \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="/etc/group:/etc/group:ro" \
	--volume="/etc/passwd:/etc/passwd:ro" \
	--volume="/etc/shadow:/etc/shadow:ro" \
	-v "$HOME:$HOME:rw"\
	--name difflocks_dock\
	-it difflocks_dock:latest

