/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#ifndef __DEEPSTREAM_TTS_APP_H_
#define __DEEPSTREAM_TTS_APP_H_

#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include "deepstream_asr_tts_config_file_parser.h"

#define CHECK_PTR(ptr) \
  if(ptr ==  NULL) \
  { \
    return -1; \
  }

#define MAX_FILENAME_LENGTH (256)
#define MUX_AUDIO_VIDEO_OUT

/* Macro for linking Gstreamer elements with source request pad and static sink pad */
#define LINK_REQUEST_PAD_STATIC_PAD(src, srcpadname, sink, sinkpadname) \
{ \
  GstPad *srcpad = gst_element_request_pad_simple (src, srcpadname); \
  if (!srcpad) { \
    g_printerr ("%s source pad %s request failed.\n", #src, #srcpadname); \
    return -1; \
  } \
  GstPad *sinkpad = gst_element_get_static_pad (sink, sinkpadname); \
  if (!sinkpad) { \
    g_printerr ("%s sink pad %s request failed.\n", #sink, #sinkpadname); \
    return -1; \
  } \
\
  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) { \
    g_printerr ("Failed to link %s to %s\n", #src, #sink); \
    return -1; \
  } \
  gst_object_unref (srcpad); \
  gst_object_unref (sinkpad); \
}

/* Macro for linking Gstreamer elements with source static pad and request sink pad */
#define LINK_STATIC_PAD_REQUEST_PAD(src, srcpadname, sink, sinkpadname) \
{ \
  GstPad *srcpad = gst_element_get_static_pad (src, srcpadname); \
  if (!srcpad) { \
    g_printerr ("%s source pad %s request failed.\n", #src, #srcpadname); \
    return -1; \
  } \
  GstPad *sinkpad = gst_element_request_pad_simple (sink, sinkpadname); \
  if (!sinkpad) { \
    g_printerr ("%s sink pad %s request failed.\n", #sink, #sinkpadname); \
    return -1; \
  } \
\
  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) { \
    g_printerr ("Failed to link %s to %s\n", #src, #sink); \
    return -1; \
  } \
  gst_object_unref (srcpad); \
  gst_object_unref (sinkpad); \
}

#define ATTACH_STATIC_PAD_PROBE(element, padname, probe_type, probe, dataptr) \
{ \
    GstPad *pad = gst_element_get_static_pad (element, padname); \
    if (!pad) { \
      g_printerr ("Get pad failed for %s %s. Exiting.\n",#element, padname); \
      return -1; \
    } \
    gst_pad_add_probe (pad, probe_type, probe, dataptr, NULL); \
    gst_object_unref (pad); \
}


typedef struct __StreamCtx
{
  gchar *uri;
  guint stream_id;
  guint has_audio;
  guint bus_id;
  GstElement *asr_pipeline;
  GstElement *input_downmixer;
  int eos_received;
  NvDsAudioConfig audio_config;
  FILE *FP_asr;
} StreamCtx;

typedef struct __AppCtx
{
  guint num_sources;
  StreamCtx *sctx;
  GstElement **proxy_audio_sinks;
  GstElement **proxy_audio_sources;
  GstElement *renderer_pipeline;
  gboolean enable_playback;
  const char *playback_output_file_name;
  gboolean pts_mode;
  gboolean sync;
  guint bus_id;
  int eos_received;
  NvDsAppConfig app_config;
} AppCtx;


int create_pipeline(AppCtx *appctx, int stream_num, StreamCtx *sctx,
    GstElement **p_proxy_audio_sink);
int create_renderer_pipeline(AppCtx *appctx);
int start_pipeline(int stream_num, StreamCtx *sctx);
int destroy_pipeline(StreamCtx *sctx);

guint get_num_sources (gchar *cfg_file_path);
gboolean parse_config_file (AppCtx *appctx, gchar *config_file);

G_BEGIN_DECLS

guint get_num_sources_yaml (gchar *cfg_file_path);
gboolean parse_config_file_yaml (AppCtx *appctx, gchar *config_file);

G_END_DECLS
#endif
