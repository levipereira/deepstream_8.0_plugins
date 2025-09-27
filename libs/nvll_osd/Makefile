#######################################################################################################
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
######################################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++
SRCS:= nvll_osd.cpp nvll_osd_int.cpp cuosd.cpp \
	textbackend/backend.cpp textbackend/pango-cairo.cpp cuda/drawBbox.cu \
	cuda/cuosd_kernel.cu

INCS:= $(wildcard *.h)
LIB:= libnvds_osd.so
PKGS:= glib-2.0 gobject-2.0
NVDS_VERSION:=8.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS:= -fPIC -Wno-deprecated-declarations -std=c++11 -DENABLE_TEXT_BACKEND_PANGO \
	 -I/usr/include/harfbuzz \
	 -I/usr/include/cairo -I/usr/include/pango-1.0 -I. \
	 -Icuda \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes -DNDEBUG

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS := -shared -Wl,-no-undefined \
	-lpthread \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lpango-1.0 -lcairo -lpangocairo-1.0 -lgobject-2.0 \
       -lnvds_utils \
       -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -lnvbufsurface -lcuda

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
