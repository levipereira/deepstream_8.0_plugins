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

#include <sys/time.h>

#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include "gstnvdsA2Vtemplate.h"
#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include <atomic>

#define DEFAULT_GPU_ON TRUE

GST_DEBUG_CATEGORY_STATIC (gst_nvdsA2Vtemplate_debug);
#define GST_CAT_DEFAULT gst_nvdsA2Vtemplate_debug

#define GST_ERROR_ON_BUS(msg, ...) do {                         \
  if(scope) {                                       \
    GST_ERROR_OBJECT(scope, __VA_ARGS__);           \
    GError *err = g_error_new(g_quark_from_static_string(       \
      GST_ELEMENT_NAME(scope)), -1, __VA_ARGS__);   \
    gst_element_post_message(GST_ELEMENT(scope),    \
      gst_message_new_error(GST_OBJECT(scope),      \
        err, msg));                                             \
  }                                                             \
}while(0)

#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1

#define GST_AUDIO_CAPS_MAKE_WITH_FEATURES(format, channels)            \
      "audio/x-raw(memory:NVMM), "                                     \
      "format = (string) " format ", "                                 \
      "rate = [ 1, 2147483647 ], "                                     \
      "layout = (string) interleaved, "                                \
      "channels = " channels

#define GST_AUDIO_SW_CAPS_MAKE_WITH_FEATURES(format, channels)           \
      "audio/x-raw, "                                                    \
      "format = (string) " format ", "                                   \
      "rate = [ 1, 2147483647 ], "                                       \
      "layout = (string) interleaved, "                                  \
      "channels = " channels
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static GstStaticPadTemplate gst_A2V_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ " "NV12, I420 }") ) );


static GstStaticPadTemplate gst_A2V_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_AUDIO_CAPS_MAKE_WITH_FEATURES("{S16LE, F32LE}", "1")
                     ";"
                     GST_AUDIO_SW_CAPS_MAKE_WITH_FEATURES("{S16LE, F32LE}", "1"))
    );

enum
{
  PROP_0,
  PROP_CUSTOMLIB_NAME,
  PROP_CUSTOMLIB_PROPS,
  PROP_CUSTOMLIB_PROPS_VALUES,
  PROP_GPU
};

static void gst_nvdsA2Vtemplate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvdsA2Vtemplate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_nvdsA2Vtemplate_finalize (GObject * object);
static gboolean gst_nvdsA2Vtemplate_decide_allocation (GstAudio2Video *scope,
    GstQuery *query);

static gboolean gst_nvdsA2Vtemplate_render (GstAudio2Video * base,
    GstBuffer * audio, GstVideoFrame * video);

#define gst_nvdsA2Vtemplate_parent_class parent_class
G_DEFINE_TYPE (GstNvDsA2Vtemplate, gst_nvdsA2Vtemplate, GST_TYPE_AUDIO2VIDEO);

static GstStateChangeReturn
gst_nvdsA2Vtemplate_change_state (GstElement *bscope, GstStateChange transition)
{
  if(transition==GST_STATE_CHANGE_NULL_TO_READY) {
    GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (bscope);
    DSCustom_CreateParams params = {0};

    bool ret;
    try {
      scope->algo_factory = new DSCustomLibrary_Factory();
      scope->algo_ctx = scope->algo_factory->CreateCustomAlgoCtx(scope->custom_lib_name, G_OBJECT(bscope));

      if(scope->algo_ctx && scope->vecProp && scope->vecProp && scope->vecProp->size()) {
          GST_INFO_OBJECT(scope, "Setting custom lib properties # %lu", scope->vecProp->size());
          for(std::vector<Property>::iterator it = scope->vecProp->begin(); it != scope->vecProp->end(); ++it) {
              GST_INFO_OBJECT(scope, "Adding Prop: %s : %s", it->key.c_str(), it->value.c_str());
              ret = scope->algo_ctx->SetProperty(*it);
              if (!ret) {
                 return GST_STATE_CHANGE_FAILURE;
              }
          }
      }
    }
    catch (const std::runtime_error& e) {
      GST_ERROR_ON_BUS("Exception occurred", "Runtime error: %s", e.what());
      return GST_STATE_CHANGE_FAILURE;
    }
    catch (...) {
      GST_ERROR_ON_BUS("Exception occurred", "Exception occurred");
      return GST_STATE_CHANGE_FAILURE;
    }
    params.m_element = bscope;

    if(!scope->algo_ctx->SetInitParams(&params)) {
      GST_ERROR_ON_BUS("SetInitParams Error", "SetInitParams Error");
      return GST_STATE_CHANGE_FAILURE;
    }
  }
  return GST_NVDSA2VTEMPLATE_GET_CLASS(bscope)->parent_change_state_fn(bscope,transition);
}

static void
gst_nvdsA2Vtemplate_class_init (GstNvDsA2VtemplateClass * g_class)
{
  GObjectClass *gobject_class = (GObjectClass *) g_class;
  GstElementClass *gstelement_class = (GstElementClass *) g_class;
  GstAudio2VideoClass *scope_class = (GstAudio2VideoClass *) g_class;

  gobject_class->set_property = gst_nvdsA2Vtemplate_set_property;
  gobject_class->get_property = gst_nvdsA2Vtemplate_get_property;
  gobject_class->finalize = gst_nvdsA2Vtemplate_finalize;
  scope_class->decide_allocation = gst_nvdsA2Vtemplate_decide_allocation;

  g_class->parent_change_state_fn = gstelement_class->change_state;
  gstelement_class->change_state =
      GST_DEBUG_FUNCPTR (gst_nvdsA2Vtemplate_change_state);

  scope_class->render = GST_DEBUG_FUNCPTR (gst_nvdsA2Vtemplate_render);

  g_object_class_install_property (gobject_class, PROP_CUSTOMLIB_NAME,
          g_param_spec_string ("customlib-name", "Custom library name",
            "Set custom library Name to be used",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CUSTOMLIB_PROPS,
          g_param_spec_string ("customlib-props", "Custom Library Properties",
            "Set Custom Library Properties (key:value) string, can be set multiple times,"
            "vector is maintained internally\n\t\t\texport NVDS_CUSTOMLIB=/path/to/customlib.so to get customlib properties",
            NULL,
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU,
          g_param_spec_boolean ("gpu-on", "GPU On/Off setting",
            "Switch between device and host memory",
            DEFAULT_GPU_ON,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  if (g_getenv ("NVDS_CUSTOMLIB")) {
      g_object_class_install_property (gobject_class, PROP_CUSTOMLIB_PROPS_VALUES,
          g_param_spec_string ("props-values",
            "Custom library propperty values",
            "Customlib property values",
            NULL,
            (GParamFlags) (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
  }

  gst_element_class_set_static_metadata (gstelement_class,
      "nvdsA2Vtemplate plugin for Audio In/Video Out use-cases",
      "Audio2Video",
      "A custom algorithm can be hooked for Audio In/Video Out use-cases",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries");

  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_A2V_src_template);
  gst_element_class_add_static_pad_template (gstelement_class,
      &gst_A2V_sink_template);

}

static gboolean
gst_nvdsA2Vtemplate_sink_event (GstPad *sinkpad, GstObject *bscope, GstEvent *event)
{
    gboolean ret = TRUE;
    GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (bscope);

    if(scope->algo_ctx) {
      ret  = scope->algo_ctx->HandleEvent(event);
      if (!ret)
        return ret;
    }

    ret = scope->parent_sink_event_fn(sinkpad, bscope, event);
    if(ret == FALSE) {
      GstState cur_state = GST_STATE_NULL;
      gst_element_get_state(GST_ELEMENT (bscope), &cur_state, NULL, 0);
      if(!(event != NULL || cur_state == GST_STATE_NULL || cur_state == GST_STATE_PAUSED))
      GST_ERROR_ON_BUS("sink_event error", "sink_event error");
    }
    return ret;
}

static gboolean gst_nvdsA2Vtemplate_decide_allocation (GstAudio2Video *bscope, GstQuery *query)
{
    GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (bscope);
    GstCaps *outcaps;
    GstStructure *config;
    GstBufferPool *pool;
    GstAllocationParams params;
    gst_allocation_params_init (&params);

      GST_DEBUG_OBJECT (scope, "Inside decide_allocation function \n");
    if(!scope->g_decide_allocation){
      scope->g_decide_allocation=1;

      if(scope->pool) {
        g_object_unref(scope->pool);
        scope->pool=NULL;
      }

      gst_query_parse_allocation (query, &outcaps, NULL);
      pool = gst_nvds_buffer_pool_new ();
      config = gst_buffer_pool_get_config (pool);

      cudaSetDevice(0);

      GST_DEBUG_OBJECT (scope, "nvdsA2Vtemplate caps = %s\n", gst_caps_to_string(outcaps));
      gst_buffer_pool_config_set_params (config, outcaps, sizeof (NvBufSurface), 4, 4); // TODO: remove 4 hardcoding

      if(scope->gpu_on) {
        gst_structure_set (config,
          "memtype", G_TYPE_UINT, NVBUF_MEM_CUDA_DEVICE,
          "gpu-id", G_TYPE_UINT, DEFAULT_GPU_ID,
          "batch-size", G_TYPE_UINT, DEFAULT_BATCH_SIZE, NULL);
      } else {
        gst_structure_set (config,
          "memtype", G_TYPE_UINT, NVBUF_MEM_CUDA_UNIFIED,
          "gpu-id", G_TYPE_UINT, DEFAULT_GPU_ID,
          "batch-size", G_TYPE_UINT, DEFAULT_BATCH_SIZE, NULL);
      }

      GST_INFO_OBJECT (scope, " %s Allocating Buffers in NVM Buffer Pool for Max_Views\n",
        __func__ );

      /* set config for the created buffer pool */
      if (!gst_buffer_pool_set_config (pool, config)) {
        GST_WARNING ("bufferpool configuration failed");
        return FALSE;
      }

      gboolean is_active = gst_buffer_pool_set_active (pool, TRUE);
      if (!is_active) {
        GST_WARNING (" Failed to allocate the buffers inside the output pool");
        return FALSE;
      } else {
        GST_DEBUG (" Output buffer pool (%p) successfully created",
        pool);
      }

      scope->pool = pool;
    }
    GST_AUDIO2VIDEO_CLASS(parent_class)->decide_allocation(bscope, query);
    return TRUE;
}

static GstPadProbeReturn
nvdsA2Vtemplate_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    if(gst_buffer_get_size(GST_BUFFER(info->data))!=sizeof(NvBufSurface))
      return GST_PAD_PROBE_DROP;

    return GST_PAD_PROBE_OK;
}

static void
gst_nvdsA2Vtemplate_init (GstNvDsA2Vtemplate * scope)
{
  scope->gpu_on = DEFAULT_GPU_ON;
  scope->g_decide_allocation = 0;
  GstPad *sinkpad = GST_PAD(GST_ELEMENT(scope)->sinkpads->data);
  GstPad *srcpad = GST_PAD(GST_ELEMENT(scope)->srcpads->data);

  scope->parent_sink_event_fn = GST_PAD_EVENTFUNC (sinkpad);

  gst_pad_set_event_function(sinkpad,
    GST_DEBUG_FUNCPTR (gst_nvdsA2Vtemplate_sink_event));

  gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BUFFER,
    nvdsA2Vtemplate_src_pad_buffer_probe, NULL, NULL);
}

static void
gst_nvdsA2Vtemplate_finalize (GObject * object)
{
  GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (object);

  if (scope->algo_ctx)
    delete scope->algo_ctx;

  if (scope->algo_factory)
    delete scope->algo_factory;

  if (scope->vecProp)
    delete scope->vecProp;

  if (scope->custom_lib_name) {
    g_free(scope->custom_lib_name);
    scope->custom_lib_name = NULL;
  }

  if (scope->custom_prop_string) {
    g_free(scope->custom_prop_string);
    scope->custom_prop_string = NULL;
  }

  if(scope->pool) {
    scope->g_decide_allocation = 0;
    g_object_unref(scope->pool);
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_nvdsA2Vtemplate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (object);

  switch (prop_id) {
    case PROP_CUSTOMLIB_NAME:
      if (scope->custom_lib_name) {
        g_free(scope->custom_lib_name);
      }
      scope->custom_lib_name = (gchar *)g_value_dup_string (value);
      break;
    case PROP_CUSTOMLIB_PROPS:
    {
      if (!scope->vecProp)
      {
        scope->vecProp = new std::vector<Property>;
      }
      {
        if (scope->custom_prop_string) {
          g_free(scope->custom_prop_string);
          scope->custom_prop_string = NULL;
        }
        scope->custom_prop_string = (gchar *)g_value_dup_string (value);
        std::string propStr (scope->custom_prop_string);
        std::size_t found = 0;
        std::size_t start = 0;

        found = propStr.find_first_of(":");
        if (found == 0)
        {
          GST_ERROR_ON_BUS("Custom Library property Error",
            "Custom Library property Error: required format is: customlib-props=\"[key:value]\"");
          return;
        }
        Property prop(propStr.substr(start, found), propStr.substr(found+1));
        scope->vecProp->push_back(prop);
        if(nullptr != scope->algo_ctx) {
          bool ret = scope->algo_ctx->SetProperty(prop);
          if(!ret) {
            GST_ERROR_ON_BUS("SetProperty Error",
              "SetProperty Error (%s:%s)", prop.key.c_str(), prop.value.c_str());
          }
        }
      }
    }
    break;
    case PROP_GPU:
      scope->gpu_on = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_nvdsA2Vtemplate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (object);

  switch (prop_id) {
    case PROP_CUSTOMLIB_NAME:
      g_value_set_string (value, scope->custom_lib_name);
      break;
    case PROP_CUSTOMLIB_PROPS:
      g_value_set_string (value, scope->custom_prop_string);
      break;
    case PROP_CUSTOMLIB_PROPS_VALUES:
      if (g_getenv("NVDS_CUSTOMLIB"))
      {
          DSCustomLibrary_Factory *algo_factory = new DSCustomLibrary_Factory();
          char *str = NULL;
          IDSCustomLibrary *algo_ctx = algo_factory->CreateCustomAlgoCtx(g_getenv("NVDS_CUSTOMLIB"), object);
          if (algo_ctx) {
              str = algo_ctx->QueryProperties ();
              delete algo_ctx;
          }
          if (algo_factory)
              delete algo_factory;

          g_value_set_string (value, str);

          if (str)
              delete str;
      }
      break;
    case PROP_GPU:
        g_value_set_boolean (value, scope->gpu_on);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_nvdsA2Vtemplate_render (GstAudio2Video * base, GstBuffer * audio,
    GstVideoFrame * video)
{
  GstElement *element = GST_ELEMENT(base);
  GstPad *srcpad = GST_PAD(element->srcpads->data);
  GstFlowReturn ret;
  BufferResult ret_process_buffer= BufferResult::Buffer_Ok;

  GstNvDsA2Vtemplate *scope = GST_NVDSA2VTEMPLATE (base);
  GstBuffer *outbuf = NULL, *temp=NULL;

  ret = gst_buffer_pool_acquire_buffer (scope->pool, &outbuf, NULL);
  if(ret!=GST_FLOW_OK) {
    GST_ERROR_ON_BUS("failed to activate bufferpool", "failed to activate bufferpool");
    return false;
  }

  GST_BUFFER_PTS (outbuf) = GST_BUFFER_PTS (video->buffer);
  GST_BUFFER_DURATION (outbuf) = GST_BUFFER_DURATION (video->buffer);

  temp = video->buffer;
  video->buffer =  outbuf;

  ret_process_buffer = scope->algo_ctx->ProcessBuffer (base, audio, video);
  if(ret_process_buffer == BufferResult::Buffer_Error){
    gst_buffer_unref(outbuf);
    video->buffer =  temp;
    GST_ERROR_ON_BUS("ProcessBuffer Error", "ProcessBuffer Error");
    return false;
  }

  if(ret_process_buffer==BufferResult::Buffer_Ok) {
  ret = gst_pad_push (srcpad, outbuf);
    if(ret!=GST_FLOW_OK) {
      video->buffer =  temp;
      GST_ERROR_ON_BUS("failed to push buffer", "failed to push buffer");
      return false;
    }
  }
  else if(ret_process_buffer==BufferResult::Buffer_Drop){
    gst_buffer_unref(outbuf);
  }

  video->buffer =  temp;
  return true;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvdsA2Vtemplate_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvdsA2Vtemplate_debug, "nvdsA2Vtemplate", 0,
      "nvdsA2Vtemplate plugin");

  return gst_element_register (plugin, "nvdsA2Vtemplate", GST_RANK_PRIMARY,
      GST_TYPE_NVDSA2VTEMPLATE);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgstA2Vtemplate,
    DESCRIPTION,
    nvdsA2Vtemplate_plugin_init, "8.0", LICENSE, BINARY_PACKAGE,
    URL)
