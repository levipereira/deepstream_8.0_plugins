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

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CC:= g++
NVCC:=/usr/local/cuda/bin/nvcc

DS_SRC_PATH:= /opt/nvidia/deepstream/deepstream
LIB_INSTALL_DIR?= /opt/nvidia/deepstream/deepstream/lib/

# Change to your deepstream SDK includes
CFLAGS+= -I$(DS_SRC_PATH)/sources/includes \
         -I/usr/local/cuda/include \
         -I$(DS_SRC_PATH)/sources/includes/nvdsinferserver \
         -I$(DS_SRC_PATH)/sources/includes/ds3d/common/helper

CFLAGS+= -Wall -std=c++14 -shared -fPIC

CUFLAGS:= -I../../includes -I/usr/local/cuda/include \
          -gencode arch=compute_61,code=compute_61 -gencode arch=compute_70,code=compute_70 \
          -gencode arch=compute_75,code=compute_75 -gencode arch=compute_80,code=compute_80 \
          -gencode arch=compute_86,code=compute_86 -gencode arch=compute_90,code=compute_90

ifeq ($(TARGET_DEVICE),aarch64)
CUFLAGS+= -gencode arch=compute_72,code=compute_72 -gencode arch=compute_87,code=compute_87
else
CUFLAGS+= -gencode arch=compute_60,code=compute_60 -gencode arch=compute_89,code=compute_89
endif

LIBS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lyaml-cpp

TARGET_LIB:= libnvds_3d_v2x_infer_custom_postprocess.so
TARGET_LIB_SRCS:= nvinferserver_custom_postprocess.cpp \
                  decoder.cu

TARGET_LIB_INCS:= $(wildcard ./*.h ./*.hpp)
TARGET_LIB_OBJS:= $(TARGET_LIB_SRCS:.cpp=.o)
TARGET_LIB_OBJS:= $(TARGET_LIB_OBJS:.cu=.o)

LDFLAGS:= -shared -Wl,--no-undefined -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB): $(TARGET_LIB_OBJS) Makefile
	$(CC) -o $(TARGET_LIB) $(TARGET_LIB_OBJS) $(LDFLAGS)

%.o: %.cpp $(TARGET_LIB_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC -I$(DS_SRC_PATH)/sources/includes/ds3d/common/helper' $(CUFLAGS) $<

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(TARGET_LIB_OBJS) $(TARGET_LIB)

