/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _GST_NVDSUDPSRC_H_
#define _GST_NVDSUDPSRC_H_

#include <gst/base/gstpushsrc.h>
#include <gio/gio.h>
#include <rivermax_api.h>

#include "gstnvdsudpcommon.h"

G_BEGIN_DECLS

#define GST_TYPE_NVDSUDPSRC   (gst_nvdsudpsrc_get_type())
#define GST_NVDSUDPSRC(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSUDPSRC,GstNvDsUdpSrc))
#define GST_NVDSUDPSRC_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSUDPSRC,GstNvDsUdpSrcClass))
#define GST_IS_NVDSUDPSRC(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSUDPSRC))
#define GST_IS_NVDSUDPSRC_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSUDPSRC))

typedef struct _GstNvDsUdpSrc GstNvDsUdpSrc;
typedef struct _GstNvDsUdpSrcClass GstNvDsUdpSrcClass;
typedef struct DstStreamInfo {
  char *ip;
  uint16_t port;
} DstStreamInfo;

struct _GstNvDsUdpSrc
{
  GstPushSrc parent;

  /* properties */
  gchar *uri;
  gchar *localIfaceIps;
  gint port;
  gint buffer_size;
  gchar *multi_iface;
  gboolean auto_multicast;
  GstCaps *caps;
  gchar *address;
  gchar *sourceAddresses;
  GSocket *used_socket;
  guint64 timeout;
  gboolean reuse;
  gboolean loop;
  guint headerSize;
  guint payloadSize;
  guint numPackets;
  guint payMultiple;
  gint gpuId;
  gboolean use_rtp_timestamp;
  gboolean adjust_leap_seconds;
  gchar *ptpSrc;
  gchar *st2022_7_streams;   /* List of IP:port pairs for ST2022-7 redundant streams */
  guint num_streams;         /* Number of active streams (primary + redundant) */
  guint num_source_addresses; /* Number of source addresses for filtering */
  guint num_local_interfaces; /* Number of local interfaces */

  StreamType streamType;
  VideoType videoType;
  MemoryType outputMemType;
  guint frameSize;
  guint stride;
  uint8_t *dataPtr1;
  uint8_t *dataPtr2;
  guint len1;
  guint len2;
  gboolean mBit;
  gboolean ffFound;
  guint packetCounter;

  GstBufferPool *pool;
  GThread *pThread;
  GQueue *bufferQ;
  GCond qCond;
  GMutex qLock;
  GCond flowCond;
  GMutex flowLock;
  gboolean isRunning;
  gboolean isPlayingState;
  gboolean flushing;
  GInetSocketAddress *addr;
  GCancellable *cancellable;
  gint cancellableFd;
  gint pollfd;
  gint lastError;
  gboolean isRtpOut;
  /*For rtpticks based buffer pts/dts generation */
  gboolean first_rtp_packet;
  guint64 last_rtp_tick;
  guint64 rtp_tick_base;
  guint clock_rate;
  gboolean is_nvmm;

  /* Rivermax specific */
  guint flowId;
  struct sockaddr_in localNicAddr;
  rmax_stream_id streamId[MAX_ST2022_7_STREAMS];
  struct rmax_in_flow_attr flowAttr[MAX_ST2022_7_STREAMS];
  struct rmax_in_buffer_attr bufferAttr;
  struct rmax_in_memblock data;
  struct rmax_in_memblock hdr;
  gboolean isGpuDirect;
  gpointer dataPtr;
  gpointer hdrPtr;
  size_t alignedMemSize;

  DstStreamInfo dstStream[MAX_ST2022_7_STREAMS];
  gchar *srcAddress[MAX_ST2022_7_STREAMS];
  gchar *localIfaceIp[MAX_ST2022_7_STREAMS];
};

struct _GstNvDsUdpSrcClass
{
  GstPushSrcClass parent_class;
};

GType gst_nvdsudpsrc_get_type (void);

G_END_DECLS

#endif
