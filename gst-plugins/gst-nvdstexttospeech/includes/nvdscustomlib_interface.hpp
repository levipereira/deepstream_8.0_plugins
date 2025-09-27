/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDS_TTS_CUSTOMLIB_INTERFACE_HPP__
#define __NVDS_TTS_CUSTOMLIB_INTERFACE_HPP__

#include <gst/base/gstbasetransform.h>
#include <gst/gstbuffer.h>
#include <string>

#define NVDS_CONFIG_FILE_PROPERTY "config-file"

namespace nvdstts {

enum class BufferResult {
    Buffer_Ok,  // Push the buffer from submit_input function
    Buffer_Drop,  // Drop the buffer inside submit_input function
    Buffer_Async,  // Return from submit_input function, custom lib to push the
                   // buffer
    Buffer_Error  // Error occured
};

struct DSCustom_CreateParams {
    GstBaseTransform* m_element;
    GstCaps* m_inCaps;
    GstCaps* m_outCaps;
};

struct Property {
    Property(std::string arg_key, std::string arg_value)
        : key(arg_key), value(arg_value)
    {
    }

    std::string key;
    std::string value;
};

enum class CapsType : int {
    kNone = 0,
    kAudio,
    kText,
};

class IDSCustomLibrary {
public:
    virtual bool SetProperty(const Property& prop) = 0;
    virtual bool Initialize() = 0;
    virtual GstCaps* GetCompatibleCaps(
        GstPadDirection direction, GstCaps* in_caps, GstCaps* othercaps) = 0;
    virtual bool StartWithParams(DSCustom_CreateParams* params) = 0;
    virtual bool HandleEvent(GstEvent* event) = 0;
    virtual BufferResult ProcessBuffer(GstBuffer* inbuf) = 0;
    virtual ~IDSCustomLibrary(){};
};

}  // namespace nvdstts

#endif
