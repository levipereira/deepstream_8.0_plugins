################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

DS_VER = $(shell deepstream-app -v | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' )

DS_SRC_PATH := /opt/nvidia/deepstream/deepstream-$(DS_VER)
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(DS_VER)/lib/
CC:= g++

# Change to your deepstream SDK includes
CFLAGS+= -I$(DS_SRC_PATH)/sources/includes \
	  -I/usr/local/cuda/include \
	  -I$(DS_SRC_PATH)/sources/includes/nvdsinferserver

CFLAGS+= -Wall -std=c++14 -shared -fPIC

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= custom_postprocess.cpp lidar_postprocess.cpp
TARGET_LIB:= libnvds_lidar_custom_postprocess_impl.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB)

