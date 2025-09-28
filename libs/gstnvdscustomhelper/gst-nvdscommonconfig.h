/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _GST_NVDSCOMMON_CONFIG_H_
#define _GST_NVDSCOMMON_CONFIG_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <gst/gst.h>

typedef enum
{
  SOURCE_TYPE_AUTO,
  SOURCE_TYPE_URI,
  SOURCE_TYPE_RTSP
} NvDsUriSrcBinType;

typedef enum
{
  DEC_SKIP_FRAMES_TYPE_NONE,
  DEC_SKIP_FRAMES_TYPE_NONREF,
  DEC_SKIP_FRAMES_TYPE_KEY_FRAME_ONLY
} NvDsUriSrcBinDecSkipFrame;

typedef enum
{
  RTP_PROTOCOL_UNKNOWN = 0,
  RTP_PROTOCOL_UDP = 1,
  RTP_PROTOCOL_UDP_MCAST = 2,
  RTP_PROTOCOL_TCP = 4,
  RTP_PROTOCOL_UDP_UDPMCAST_TCP = 7,
  RTP_PROTOCOL_HTTP = 10,
  RTP_PROTOCOL_TLS = 20
} NvDsUriSrcBinRtpProtocol;

typedef enum
{
  LEAKY_NONE,
  LEAKY_UPSTREAM,
  LEAKY_DOWNSTREAM
} NvDsUriSrcBinLeaky;

typedef enum
{
  BUFFER_MODE_UNKNOWN = 0,
  BUFFER_MODE_SLAVE = 1,
  BUFFER_MODE_BUFFER = 2,
  BUFFER_MODE_AUTO = 3,
  BUFFER_MODE_SYNCED = 4
} NvDsUriSrcBinBufferMode;

typedef enum
{
  SMART_REC_DISABLE,
  SMART_REC_CLOUD,
  SMART_REC_MULTI
} NvDsUriSrcBinSRType;

typedef enum
{
  SMART_REC_AUDIO_VIDEO,
  SMART_REC_VIDEO_ONLY,
  SMART_REC_AUDIO_ONLY
} NvDsUriSrcBinSRMode;

typedef enum
{
  SMART_REC_MP4,
  SMART_REC_MKV
} NvDsUriSrcBinSRCont;

typedef struct _NvDsSensorInfo
{
  guint source_id;
  gchar const* uri;
  gchar const* sensor_id;
  gchar const* sensor_name;
}NvDsSensorInfo;

typedef struct _NvDsRtspAttemptsInfo
{
  gboolean attempt_exceeded;
  guint source_id;
}NvDsRtspAttemptsInfo;

typedef struct _GstDsNvUriSrcConfig
{
  NvDsUriSrcBinType src_type;
  gboolean loop;
  gchar *uri;
  gchar *sei_uuid;
  gint latency;
  NvDsUriSrcBinSRType smart_record;
  gchar *smart_rec_dir_path;
  gchar *smart_rec_file_prefix;
  NvDsUriSrcBinSRCont smart_rec_container;
  NvDsUriSrcBinSRMode smart_rec_mode;
  guint smart_rec_def_duration;
  guint smart_rec_cache_size;
  guint gpu_id;
  gint source_id;
  NvDsUriSrcBinRtpProtocol rtp_protocol;
  NvDsUriSrcBinBufferMode buffer_mode;
  guint num_extra_surfaces;
  NvDsUriSrcBinDecSkipFrame skip_frames_type;
  guint cuda_memory_type;
  guint drop_frame_interval;
  gboolean low_latency_mode;
  gboolean extract_sei_type5_data;
  gint rtsp_reconnect_interval_sec_org;
  gint rtsp_reconnect_interval_sec;
  NvDsUriSrcBinLeaky leaky;
  guint max_size_buffers;
  gint init_rtsp_reconnect_interval_sec;
  gint rtsp_reconnect_attempts;
  gint num_rtsp_reconnects;
  guint udp_buffer_size;
  gchar *sensorId; /**< unique Sensor ID string */
  gboolean disable_passthrough;
  gchar *sensorName; /**< Sensor Name string; could be NULL */
  gboolean disable_audio;
  gboolean drop_on_latency;
  gboolean ipc_buffer_timestamp_copy;
  gchar *ipc_socket_path;
  gint ipc_connection_attempts;
  guint64 ipc_connection_interval;
  gboolean sensorIdToPadIdMapping;
} GstDsNvUriSrcConfig;

typedef struct
{
  // Struct members to store config / properties for the element

  //mandatory configs when using legacy nvstreammux
  gint pipeline_width;
  gint pipeline_height;
  gint batched_push_timeout;

  //not mandatory; auto configured
  gint batch_size;

  //not mandatory; defaults will be used
  gint buffer_pool_size;
  gint compute_hw;
  gint num_surfaces_per_frame;
  gint interpolation_method;
  guint gpu_id;
  guint nvbuf_memory_type;
  gboolean live_source;
  gboolean enable_padding;
  gboolean attach_sys_ts_as_ntp;
  gchar* config_file_path;
  gboolean sync_inputs;
  guint64 max_latency;
  gboolean frame_num_reset_on_eos;
  gboolean frame_num_reset_on_stream_reset;

  guint64 frame_duration;
  guint maxBatchSize;
  gboolean async_process;
  gboolean no_pipeline_eos;
  gboolean extract_sei_type5_data;
  gboolean sort_batch;
  gboolean buffer_cache;
  gint buffer_cache_timeout;
  gboolean extract_sei_sim_time;
  gboolean align_first_buffer;
  guint sync_inputs_ntp;
} GstDsNvStreammuxConfig;

#ifdef __cplusplus
}
#endif

#endif
