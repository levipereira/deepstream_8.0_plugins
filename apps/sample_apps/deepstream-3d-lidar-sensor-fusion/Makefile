################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

NVDS_DIR := $(shell realpath /opt/nvidia/deepstream/deepstream)

LIB_INSTALL_DIR?=$(NVDS_DIR)/lib
SRC_INSTALL_DIR?=$(NVDS_DIR)/sources
APP_INSTALL_DIR?=$(NVDS_DIR)/bin

APP:= deepstream-3d-lidar-sensor-fusion

CC = g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CFLAGS:= -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.hpp)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I$(SRC_INSTALL_DIR)/includes \
         -I$(SRC_INSTALL_DIR)/libs \
         -I /usr/local/cuda/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda/lib64/ -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_3d_gst \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl -pthread
APP_OBJS:= $(APP_SRCS:.cpp=.o)

debug: $(APP)
all: $(APP)

%.o: %.cpp $(APP_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(APP_OBJS) Makefile
	$(CC) -o $(APP) $(APP_OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)/

clean:
	rm -rf $(APP_OBJS) $(APP)

.PHONY: all
