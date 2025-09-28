# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

PKGS:= glib-2.0 gobject-2.0 json-glib-1.0 uuid
PKGS += gstreamer-1.0 gstreamer-base-1.0 gstreamer-audio-1.0 gstreamer-video-1.0
TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++

SRCS:= $(wildcard *.c)
SRCS += $(wildcard *.cpp)

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_multistream.so

NVDS_VERSION:=8.0

CFLAGS+= -fPIC -DDS_VERSION=\"8.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes \
	 -I ../../libs/nvstreammux/include \
	 -std=c++17 -Wall -Werror

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
	LIBS := -L/usr/lib/aarch64-linux-gnu/tegra
endif

LIBS+= -shared -Wl,-no-undefined \
	-L$(GST_INSTALL_DIR) -lnvdsgst_multistream_legacy \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl -lpthread \
	-L../../libs/nvstreammux/ -lnvds_streammux \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lgstnvdsseimeta \
	-lyaml-cpp -lm \
	-lnvds_nvtxhelper -lnvdsbufferpool\
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
