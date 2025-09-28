/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVDSCUSTOMLIB_FACTORY_HPP__
#define __NVDSCUSTOMLIB_FACTORY_HPP__

#include <dlfcn.h>
#include <errno.h>

#include <iostream>
#include <functional>

#include "nvdscustomlib_interface.hpp"

template<class T>
T* dlsym_ptr(void* handle, char const* name) {
  return reinterpret_cast<T*>(dlsym(handle, name));
}

class DSCustomLibrary_Factory
{
public:
    DSCustomLibrary_Factory()
    {
    }

    ~DSCustomLibrary_Factory()
    {
        if (m_libHandle)
        {
            dlclose(m_libHandle);
            m_libHandle = NULL;
            m_libName.clear();
        }
    }

    IDSCustomLibrary *CreateCustomAlgoCtx(std::string libName, GObject* object)
    {
        m_libName.assign(libName);

        m_libHandle = dlopen(m_libName.c_str(), RTLD_NOW);
        std::function<IDSCustomLibrary*(GObject*)> createAlgoCtx = nullptr;
        if (m_libHandle)
        {
            //std::cout << "Library Opened Successfully" << std::endl;

            createAlgoCtx = dlsym_ptr<IDSCustomLibrary*(GObject*)>(m_libHandle, "CreateCustomAlgoCtx");
            if (!createAlgoCtx)
            {
                throw std::runtime_error("createCustomAlgoCtx function not found in library");
            }
        }
        else
        {
            throw std::runtime_error(dlerror());
        }

        return createAlgoCtx ? createAlgoCtx(object) : nullptr;
    }

public:
    void *m_libHandle;
    std::string m_libName;
};

#endif
