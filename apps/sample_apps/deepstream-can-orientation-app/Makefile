################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NVDS_VERSION:=8.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/
SOURCE_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/sources/

DS_CAN_CONTEXT_LIB:= libnvds_can_context.a
APP:= deepstream-can-orientation-app

CC = g++
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CFLAGS:= -g -O0 -fvisibility=hidden -Wall -Werror
debug: CFLAGS+= -g -O0

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS+= -DPLATFORM_TEGRA
endif

APP_SRCS:= deepstream_can_main.cpp
DS_CAN_CONTEXT_SRCS:= deepstream_can_context.cpp
APP_INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 yaml-cpp

CFLAGS+= -I$(SOURCE_DIR)/includes \
         -I$(SOURCE_DIR)/libs \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -fPIC -std=c++14

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lnvdsgst_helper -lm \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_3d_gst \
		-L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
		-lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
		-lgstapp-1.0 -ldl -lcuda -lcudart -pthread

APP_OBJS:= $(APP_SRCS:.cpp=.o)
DS_CAN_CONTEXT_OBJS:= $(DS_CAN_CONTEXT_SRCS:.cpp=.o)

debug: $(APP)
all: $(APP)

%.o: %.cpp $(APP_INCS)
	$(CC) -c -o $@ $(CFLAGS) $<

$(DS_CAN_CONTEXT_LIB): $(DS_CAN_CONTEXT_OBJS)
	$(AR) rcs $@ $(DS_CAN_CONTEXT_OBJS)
	ranlib $@

$(APP): $(APP_OBJS) $(DS_CAN_CONTEXT_LIB)
	$(CC) -o $(APP) $(APP_OBJS) -L. -l:$(DS_CAN_CONTEXT_LIB) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(APP_OBJS) $(APP) $(DS_CAN_CONTEXT_LIB)

.PHONY: all
