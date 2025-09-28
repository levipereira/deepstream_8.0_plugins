################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

NVDS_VERSION := $(shell deepstream-app -v | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' )

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

APP:= deepstream-lidar-inference-app

CC = g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
DS_VER = $(shell deepstream-app -v | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' )
DS_SRC_PATH := /opt/nvidia/deepstream/deepstream-$(DS_VER)

CFLAGS:= -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

APP_SRCS:= $(wildcard *.cpp)
APP_INCS:= $(wildcard *.hpp)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I$(DS_SRC_PATH)/sources/includes \
         -I$(DS_SRC_PATH)/sources/libs \
         -I$(DS_SRC_PATH)/sources/includes/nvdsinferserver \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+=  -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_3d_gst \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl
APP_OBJS:= $(APP_SRCS:.cpp=.o)

SUBFOLDERS:=custom_postprocess_impl custom_preprocess_impl

all: $(APP) $(SUBFOLDERS)
%.o: %.cpp $(APP_INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(APP_OBJS) Makefile
	$(CC) -o $(APP) $(APP_OBJS) $(LIBS)

$(SUBFOLDERS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

install: $(APP) $(SUBFOLDERS)
	cp -rv $(APP) $(APP_INSTALL_DIR)
	cp -rv custom_postprocess_impl/libnvds_lidar_custom_postprocess_impl.so $(LIB_INSTALL_DIR)
	cp -rv custom_preprocess_impl/libnvds_lidar_custom_preprocess_impl.so $(LIB_INSTALL_DIR)

clean: $(SUBFOLDERS)
	rm -rf $(APP_OBJS) $(APP)

.PHONY: all $(SUBFOLDERS)