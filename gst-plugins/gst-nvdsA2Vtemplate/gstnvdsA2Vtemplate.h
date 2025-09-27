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

#ifndef __GST_NVDSA2VTEMPLATE_H__
#define __GST_NVDSA2VTEMPLATE_H__

#include <vector>
#include "gstaudio2video.h"
#include "nvdscustomlib_factory.hpp"
#include "nvdscustomlib_interface.hpp"

G_BEGIN_DECLS

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsA2Vtemplate"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "Nvidia nvdsA2Vtemplate plugin for handling audio in to video out related use cases \0"
#define BINARY_PACKAGE "NVIDIA DeepStream A2Vtemplate Plugin"
#define URL "http://nvidia.com/"

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSA2VTEMPLATE            (gst_nvdsA2Vtemplate_get_type())
#define GST_NVDSA2VTEMPLATE(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSA2VTEMPLATE,GstNvDsA2Vtemplate))
#define GST_NVDSA2VTEMPLATE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSA2VTEMPLATE,GstNvDsA2VtemplateClass))
#define GST_NVDSA2VTEMPLATE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSA2VTEMPLATE, GstNvDsA2VtemplateClass))
#define GST_IS_NVDSA2VTEMPLATE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSA2VTEMPLATE))
#define GST_IS_NVDSA2VTEMPLATE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSA2VTEMPLATE))
#define GST_NVDSA2VTEMPLATE_CAST(obj)       ((GstNvDsA2Vtemplate *)(obj))

typedef struct _GstNvDsA2Vtemplate GstNvDsA2Vtemplate;
typedef struct _GstNvDsA2VtemplateClass GstNvDsA2VtemplateClass;

struct _GstNvDsA2Vtemplate
{
  GstAudio2Video parent;
  GstPadEventFunction parent_sink_event_fn;
  GstPadQueryFunction parent_src_query_fn;

  /** Custom Library Factory and Interface */
  DSCustomLibrary_Factory *algo_factory;
  IDSCustomLibrary *algo_ctx;

  /** Custom Library Name and output caps string */
  gchar* custom_lib_name;

  /* Store custom lib property values */
  std::vector<Property> *vecProp;
  gchar *custom_prop_string;

  gboolean gpu_on;

  /* < private > */
  GstBufferPool *pool;
  gboolean g_decide_allocation;
};

struct _GstNvDsA2VtemplateClass
{
  GstAudio2VideoClass parent_class;
  GstStateChangeReturn (* parent_change_state_fn) (GstElement * element,
    GstStateChange transition);
};

GType gst_nvdsA2Vtemplate_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSA2VTEMPLATE_H__ */
