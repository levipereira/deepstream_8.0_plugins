#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

# usage bevfusion/docker_run_ds3d_sensor_fusion_bevfusion_pipeline.sh {CONFIGFILE}

set -e

# debug
# set -x

version=$(basename $(realpath /opt/nvidia/deepstream/deepstream) | awk -F'-' '{print $2}')
DEFAULT_TARGET_IMAGE="deepstream-triton-bevfusion:${version}"
DEFAULT_CONFIG_FILE="ds3d_lidar_plus_multi_cam_bev_fusion.yaml"
TARGET_DEVICE=$(uname -m)

function usage() {
    echo "usage: bevfusion/docker_run_ds3d_sensor_fusion_bevfusion_pipeline.sh {CONFIGFILE}"
    echo "CONFIGFILE, optional, default: ${DEFAULT_CONFIG_FILE}"
    echo "DOCKER_IMAGE: optional, default: ${DEFAULT_TARGET_IMAGE}, it's built by"
    echo "bevfusion/docker_build_bevfusion_image.sh"
}

TARGET_IMAGE=${DEFAULT_TARGET_IMAGE}
TARGET_WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion
CONFIG_FILE="ds3d_lidar_plus_multi_cam_bev_fusion.yaml"

if [[ $# -ge 1 ]]; then
    CONFIG_FILE=$1
else
    usage
    exit 1
fi

if [[ $# -ge 2 ]]; then
    TARGET_IMAGE=$2
fi

APP="/opt/nvidia/deepstream/deepstream/bin/deepstream-3d-lidar-sensor-fusion"
APP_CMD="-c ${CONFIG_FILE}"

DATASET_URL="https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/raw/DS_7.0/deepstream-3d-sensor-fusion/data/nuscene.tar.gz"

if [ -n "$NUSCENE_DATASET_URL" ]; then
    DATASET_URL="$NUSCENE_DATASET_URL"
fi

LIDAR_FILE="data/nuscene/LIDAR_TOP/000000-LIDAR_TOP.bin"
echo "checking nuscenes dataset, checking the first file: ${LIDAR_FILE}"
if [ ! -e ${LIDAR_FILE} ]; then
    echo "downloading dataset from ${DATASET_URL}"
    mkdir -p data || ( echo "create directory data failed"; exit -1)
    curl -o data/nuscene.tar.gz -L ${DATASET_URL} || (echo "download dataset failed."; exit 1)
    tar -pxvf data/nuscene.tar.gz -C data/ || (echo "uncompress nuscene data failed."; exit 1)
fi

[ -z "$DISPLAY" ] && (echo "Please export correct DISPLAY before running the pipeline."; exit -1)
xhost +

MOUNT_OPTIONS="-v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "

# Mount whole deepstream-3d-lidar-sensor-fusion directory to target directory
# Note: enable if users want to validate local code/config changes
#MOUNT_OPTIONS+=" -v ./:${TARGET_WORKSPACE}"

# mount local sample data for test
MOUNT_OPTIONS+=" -v ./data:${TARGET_WORKSPACE}/data"

DOCKER_GPU_ARG="--gpus all"
if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    DOCKER_GPU_ARG="--gpus all"
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    DOCKER_GPU_ARG="--runtime nvidia"
else
    echo "Unsupported platform ${TARGET_DEVICE}"
    exit -1
fi

echo "Starting deepstream-3d-lidar-sensor-fusion pipeline for bevfusion."
docker run \
    ${DOCKER_GPU_ARG} --rm --net=host --ipc=host \
    ${MOUNT_OPTIONS} \
    -w ${TARGET_WORKSPACE} \
    -e DISPLAY=$DISPLAY \
    --sig-proxy \
    --privileged \
    --entrypoint ${APP} \
    ${TARGET_IMAGE} \
    ${APP_CMD}
