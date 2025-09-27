/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSCUSTOMLIB_BASE_HPP__
#define __NVDSCUSTOMLIB_BASE_HPP__

#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

/* Buffer Pool Configuration Parameters */
struct BufferPoolConfig {
  gint cuda_mem_type;
  guint gpu_id;
  guint max_buffers;
  gint batch_size;
};

class DSCustomLibraryBase : public IDSCustomLibrary
{
public:
    explicit DSCustomLibraryBase(GstElement* bscope = nullptr);

    /* Set Init Parameters */
    virtual bool SetInitParams(DSCustom_CreateParams *params);

    virtual ~DSCustomLibraryBase();

    /* Set Custom Properties  of the library */
    virtual bool SetProperty(Property &prop) = 0;

    virtual bool HandleEvent (GstEvent *event) = 0;
    // TODO: Add getProperty as well

    /* By default, return Query_Not_Handled to allow handling by parent elemtn */
    virtual QueryResult HandleQuery(GstQuery *query, GstStructure *query_metadata) override { 
        return QueryResult::Query_Not_Handled; }

    virtual char* QueryProperties () = 0;

    virtual BufferResult ProcessBuffer(GstAudio2Video * base, GstBuffer * audio, GstVideoFrame * video) = 0;

public:
    /* Gstreamer dsexaple2 plugin's base class reference */
    GstElement *m_element;

    /** GPU ID on which we expect to execute the algorithm */
};


DSCustomLibraryBase::DSCustomLibraryBase(GstElement* bscope) : m_element(bscope) {
}

bool DSCustomLibraryBase::SetInitParams(DSCustom_CreateParams *params) {
    m_element = params->m_element;

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase() {
}

#endif
