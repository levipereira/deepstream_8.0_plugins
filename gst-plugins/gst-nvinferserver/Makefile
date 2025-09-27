################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#################################################################################

# External dependencies, update as required.
TRITON_GRPC_CLIENT ?= /opt/tritonclient
PROTOBUF_BIN_DIR ?= /opt/proto/bin

LIB := libnvdsgst_inferserver.so
NVDS_VERSION := 8.0

CUDA_VER ?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX := g++

SRCS := \
    gstnvinferserver_meta_utils.cpp \
    gstnvinferserver.cpp \
    gstnvinferserver_impl.cpp \
    $(NULL)

ifeq ($(shell uname -m), aarch64)
SRCS +=   \
     ../../libs/nvdsinferserver/infer_utils.cpp \
     nvdsinferserver_config.pb.cc \
     nvdsinferserver_plugin.pb.cc \
     nvdsinferserver_common.pb.cc
endif


INCS += $(wildcard *.h)

CFLAGS+= \
    -O3 \
    -DNDEBUG \
    -fPIC \
    -std=c++17 \
    -DDS_VERSION=\"8.0.0\" \
    -DFOR_PRIVATE \
    $(NULL)

INC_PATHS := \
    $(TRITON_GRPC_CLIENT)/include \
    /usr/local/cuda-$(CUDA_VER)/include \
    ../../includes \
    ../../includes/nvdsinferserver \
    ../../libs/nvdsinferserver \
    ../gst-nvdspreprocess/include \
    . \
    $(NULL)

CFLAGS += $(addprefix -I, $(INC_PATHS))

GST_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LDFLAGS := \
    -shared \
    -Wl,-no-undefined \
    $(NULL)

LIBS := \
    -L$(TRITON_GRPC_CLIENT)/lib \
    -l:libprotobuf.a \
    -L$(LIB_INSTALL_DIR) \
    -lnvdsgst_helper \
    -lnvdsgst_meta \
    -lnvds_meta \
    -lnvds_infer_server \
    -lnvbufsurface \
    -lnvbufsurftransform \
    -L/usr/local/cuda-$(CUDA_VER)/lib64/ \
    -lcudart \
    -ldl \
    -lpthread \
    -Wl,-rpath,$(LIB_INSTALL_DIR) \
    -lnvdsgst_customhelper \
    $(NULL)

ifeq ($(shell uname -m), aarch64)
# ARM64/Jetson specific libraries
CFLAGS += -DIS_TEGRA
LIBS += -L/usr/lib/aarch64-linux-gnu/nvidia -lcuda
endif

OBJS := $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)

PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS += $(shell pkg-config --cflags $(PKGS))
LIBS += $(shell pkg-config --libs $(PKGS))

INFERSERVER_PROTOS := $(wildcard ../../includes/*.proto)

ALL_PROTOS := $(INFERSERVER_PROTOS)
ALL_PROTOS_PBCC := $(notdir $(ALL_PROTOS:.proto=.pb.cc))
ALL_PROTOS_PBH := $(notdir $(ALL_PROTOS:.proto=.pb.h))

.PRECIOUS:  $(ALL_PROTOS_PBCC)
.PRECIOUS:  $(ALL_PROTOS_PBH)


INCS += \
    $(INFERSERVER_PROTOS) \
    $(ALL_PROTOS_PBCC) \
    $(ALL_PROTOS_PBH) \
    $(NULL)


all: $(LIB)

%.o: %.cc $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(INCS) $(DEP) Makefile
	$(CXX) -o $@ $(LDFLAGS) $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB) $(ALL_PROTOS_PBH)  $(ALL_PROTOS_PBCC)


%.pb.h %.pb.cc: ../../includes/%.proto $(INFERSERVER_PROTOS) Makefile
	@mkdir -p $(dir $@)
	$(PROTOBUF_BIN_DIR)/protoc --proto_path=../../includes --cpp_out=. $<
