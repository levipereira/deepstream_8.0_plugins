#######################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
#######################################################################################################
# this Makefile is to be used to build the Azure device client protocol adaptor .so
CXX:=g++

PKGS:= glib-2.0 

DS_LIB:=/opt/nvidia/deepstream/deepstream/lib/

SRCS:=  azure_device_client.cpp ../../nvds_msgapi_common_src/nvds_utils.cpp
TARGET_LIB:= libnvds_azure_proto.so

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../../includes
AZURE_INC:=/usr/local/include/azureiot \
		   /usr/local/include/azure_c_shared_utility \
		   /usr/local/include/azure_macro_utils \
		   /usr/local/include/umock_c

INC_PATHS:= -I $(DS_INC) $(foreach d, $(AZURE_INC), -I$d)
CFLAGS+= $(INC_PATHS)

LIBS+= -L/$(DS_LIB) -lnvds_logger -liothub_client -lpthread -lm -lssl -lcurl -lcrypto -lglib-2.0
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(DS_LIB)

clean:
	rm -rf $(TARGET_LIB)