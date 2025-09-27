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

#ifndef __NVDSCUSTOMLIB_INTERFACE_HPP__
#define __NVDSCUSTOMLIB_INTERFACE_HPP__

#include <string>
#include <gst/gstbuffer.h>
#include <cuda_runtime.h>
#include "../gstaudio2video.h"
#include <gst/video/video.h>

enum class BufferResult {
    Buffer_Ok,      // Push the buffer from submit_input function
    Buffer_Drop,    // Drop the buffer inside submit_input function
    Buffer_Async,   // Return from submit_input function, custom lib to push the buffer
    Buffer_Error    // Error occured
};

enum class QueryResult {
    Query_Ok,          // Query is handled by custom lib Successfully.
    Query_Not_Handled, // Query is not handled by custom lib and expect to get it handled by parent element
    Query_Error        // Query is handled by custom lib with Error.
};

struct DSCustom_CreateParams {
    GstElement *m_element;
};

struct Property
{
  Property(std::string arg_key, std::string arg_value) : key(arg_key), value(arg_value)
  {
  }

  std::string key;
  std::string value;
};

class IDSCustomLibrary
{
public:
    virtual bool SetInitParams (DSCustom_CreateParams *params) = 0;
    virtual bool SetProperty (Property &prop) = 0;
    virtual bool HandleEvent (GstEvent *event) = 0;
    virtual QueryResult HandleQuery(GstQuery *query, GstStructure *query_metadata) = 0;
    virtual BufferResult ProcessBuffer(GstAudio2Video * base, GstBuffer * audio, GstVideoFrame * video) = 0;
    virtual char* QueryProperties () = 0;
    virtual ~IDSCustomLibrary() {};
};

#endif
