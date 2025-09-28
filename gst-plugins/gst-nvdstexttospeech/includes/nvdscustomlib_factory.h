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

#ifndef __NVDS_TTS_CUSTOMLIB_FACTORY_HPP__
#define __NVDS_TTS_CUSTOMLIB_FACTORY_HPP__

#include <dlfcn.h>
#include <errno.h>

#include <functional>
#include <iostream>

#include "nvdscustomlib_interface.hpp"

namespace nvdstts {

template <class T>
T*
dlsym_ptr(void* handle, char const* name)
{
    return reinterpret_cast<T*>(dlsym(handle, name));
}

class DSCustomLibrary_Factory {
public:
    DSCustomLibrary_Factory() = default;

    ~DSCustomLibrary_Factory()
    {
        if (m_libHandle) {
            dlclose(m_libHandle);
        }
    }

    IDSCustomLibrary* CreateCustomAlgoCtx(
        const std::string& libName, const std::string& symName)
    {
        m_libName.assign(libName);

        // Usiing RTLD_GLOBAL to avoid libprotobuf.so 'file already exists in
        // database' error when using common .proto file in two plugins.
        // For e.g. riva_audio.proto in Riva ASR and TTS services.
        m_libHandle = dlopen(m_libName.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (m_libHandle) {
            std::cout << "Library Opened Successfully" << std::endl;

            m_CreateAlgoCtx =
                dlsym_ptr<IDSCustomLibrary*()>(m_libHandle, symName.c_str());
            if (!m_CreateAlgoCtx) {
                throw std::runtime_error(
                    "createCustomAlgoCtx function not found in library");
            }
        } else {
            throw std::runtime_error(dlerror());
        }

        return m_CreateAlgoCtx();
    }

public:
    void* m_libHandle;
    std::string m_libName;
    std::function<IDSCustomLibrary*()> m_CreateAlgoCtx;
};

}  // namespace nvdstts

#endif
