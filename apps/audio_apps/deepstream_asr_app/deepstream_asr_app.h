/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __DEEPSTREAM_ASR_APP_H_
#define __DEEPSTREAM_ASR_APP_H_

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include "deepstream_asr_config_file_parser.h"

#define CHECK_PTR(ptr) \
  if(ptr ==  NULL) \
  { \
    return -1; \
  }

typedef struct __StreamCtx
{
  gchar *uri;
  guint stream_id;
  guint has_audio;
  guint bus_id;
  GstElement *asr_pipeline;
  int eos_received;
  NvDsAudioConfig audio_config;
  FILE *FP_asr;
} StreamCtx;

typedef struct __AppCtx
{
  guint num_sources;
  StreamCtx *sctx;
  NvDsAppConfig app_config;
} AppCtx;


int create_pipeline(AppCtx *appctx, int stream_num, StreamCtx *sctx);
int start_pipeline(int stream_num, StreamCtx *sctx);
int destroy_pipeline(StreamCtx *sctx);

guint get_num_sources (gchar *cfg_file_path);
gboolean parse_config_file (AppCtx *appctx, gchar *config_file);

G_BEGIN_DECLS

guint get_num_sources_yaml (gchar *cfg_file_path);
gboolean parse_config_file_yaml (AppCtx *appctx, gchar *config_file);

G_END_DECLS
#endif
