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

#ifndef __GST_AUDIO2VIDEO_H__
#define __GST_AUDIO2VIDEO_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <gst/video/video.h>
#include <gst/audio/audio.h>
#include <gst/base/gstadapter.h>
#include <gst/pbutils/pbutils-prelude.h>

G_BEGIN_DECLS
#define GST_TYPE_AUDIO2VIDEO            (gst_audio2video_get_type())
#define GST_AUDIO2VIDEO(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_AUDIO2VIDEO,GstAudio2Video))
#define GST_AUDIO2VIDEO_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_AUDIO2VIDEO,GstAudio2VideoClass))
#define GST_AUDIO2VIDEO_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),GST_TYPE_AUDIO2VIDEO,GstAudio2VideoClass))
typedef struct _GstAudio2Video GstAudio2Video;
typedef struct _GstAudio2VideoClass GstAudio2VideoClass;
typedef struct _GstAudio2VideoPrivate GstAudio2VideoPrivate;

struct _GstAudio2Video
{
  GstElement parent;

  guint req_spf;                /* min samples per frame wanted by the subclass */

  /* video state */
  GstVideoInfo vinfo;

  /* audio state */
  GstAudioInfo ainfo;

  /*< private >*/
  GstAudio2VideoPrivate *priv;
};

struct _GstAudio2VideoClass
{
  /*< private >*/
  GstElementClass parent_class;

  /*< public >*/
  /* virtual function, called whenever the format changes */
  gboolean (*setup) (GstAudio2Video * scope);

  /* virtual function for rendering a frame */
  gboolean (*render) (GstAudio2Video * scope, GstBuffer * audio, GstVideoFrame * video);

  gboolean (*decide_allocation)   (GstAudio2Video * scope, GstQuery *query);

  /* get latency of GstAudio2Video. It can be used while responding to latency
   * query at child class. */
  GstClockTime (*get_latency) (GstAudio2Video * scope);
};

GST_PBUTILS_API
GType gst_audio2video_get_type (void);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GstAudio2Video, gst_object_unref)

G_END_DECLS
#endif /* __GST_AUDIO2VIDEO_H__ */
