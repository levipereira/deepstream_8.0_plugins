################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

DS_SRC_PATH := /opt/nvidia/deepstream/deepstream
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream/lib/
CC:= g++

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

# Change to your deepstream SDK includes
CFLAGS+= -I$(DS_SRC_PATH)/sources/includes \
         -I/usr/local/cuda/include \
         -I$(DS_SRC_PATH)/sources/includes/nvdsinferserver \
         -I$(DS_SRC_PATH)/sources/libs/nvdsinferserver \
         -I$(DS_SRC_PATH)/sources/includes \
         -I$(DS_SRC_PATH)/sources/includes/ds3d \
         -I$(DS_SRC_PATH)/sources/includes/ds3d/common \
         -I$(DS_SRC_PATH)/sources/includes/ds3d/common/helper

CFLAGS+= -Wall -std=c++17 -shared -fPIC

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lyaml-cpp

TARGET_LIB:= libnvds_3d_lidar_preprocess_datafilter.so
TARGET_LIB_SRCS:= lidar_preprocess_filter_impl.cpp    \
                  voxelization.cu                     \
                  $(DS_SRC_PATH)/sources/libs/nvdsinferserver/infer_utils.cpp \
                  $(DS_SRC_PATH)/sources/libs/nvdsinferserver/infer_cuda_utils.cpp

TARGET_LIB_INCS:= $(wildcard ./*.h)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)
TARGET_LIB_OBJS:= $(TARGET_LIB_OBJS:.cu=.o)

LDFLAGS:= -shared -Wl,--no-undefined -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LDFLAGS)

%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I$(DS_SRC_PATH)/sources/includes/ds3d/common/helper' $<

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

