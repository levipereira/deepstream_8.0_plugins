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

#include <string.h>

#include <gst/video/video.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/gstvideopool.h>

#include "gstaudio2video.h"
#include  <gst/pbutils/pbutils-enumtypes.h>

GST_DEBUG_CATEGORY_STATIC (audio2video_debug);
#define GST_CAT_DEFAULT (audio2video_debug)

static GstBaseTransformClass *parent_class = NULL;
static gint private_offset = 0;

static void gst_audio2video_class_init (GstAudio2VideoClass * klass);
static void gst_audio2video_init (GstAudio2Video * scope,
    GstAudio2VideoClass * g_class);
static void gst_audio2video_dispose (GObject * object);

static gboolean gst_audio2video_src_negotiate (GstAudio2Video * scope);
static gboolean gst_audio2video_src_setcaps (GstAudio2Video *
    scope, GstCaps * caps);
static gboolean gst_audio2video_sink_setcaps (GstAudio2Video *
    scope, GstCaps * caps);

static GstFlowReturn gst_audio2video_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buffer);

static gboolean gst_audio2video_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_audio2video_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);

static gboolean gst_audio2video_src_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static GstClockTime gst_audio2video_get_latency(GstAudio2Video * scope);

static GstStateChangeReturn gst_audio2video_change_state (GstElement *
    element, GstStateChange transition);

static gboolean gst_audio2video_do_bufferpool (GstAudio2Video * scope,
    GstCaps * outcaps);

static gboolean
default_decide_allocation (GstAudio2Video * scope, GstQuery * query);

struct _GstAudio2VideoPrivate
{
  gboolean negotiated;

  GstBufferPool *pool;
  gboolean pool_active;
  GstAllocator *allocator;
  GstAllocationParams params;
  GstQuery *query;

  /* pads */
  GstPad *srcpad, *sinkpad;

  GstAdapter *adapter;

  GstBuffer *inbuf;
  GstBuffer *tempbuf;
  GstVideoFrame tempframe;

  guint spf;                    /* samples per video frame */
  guint64 frame_duration;

  /* QoS stuff *//* with LOCK */
  gdouble proportion;
  GstClockTime earliest_time;

  guint dropped;                /* frames dropped / not dropped */
  guint processed;

  /* configuration mutex */
  GMutex config_lock;

  GstSegment segment;
};

/* base class */

GType
gst_audio2video_get_type (void)
{
  static gsize audio2video_type = 0;

  if (g_once_init_enter (&audio2video_type)) {
    static const GTypeInfo audio2video_info = {
      sizeof (GstAudio2VideoClass),
      NULL,
      NULL,
      (GClassInitFunc) gst_audio2video_class_init,
      NULL,
      NULL,
      sizeof (GstAudio2Video),
      0,
      (GInstanceInitFunc) gst_audio2video_init,
    };
    GType _type;

    /* TODO: rename when exporting it as a library */
    _type = g_type_register_static (GST_TYPE_ELEMENT,
        "GstAudio2Video", &audio2video_info, G_TYPE_FLAG_ABSTRACT);

    private_offset =
        g_type_add_instance_private (_type, sizeof (GstAudio2VideoPrivate));

    g_once_init_leave (&audio2video_type, _type);
  }
  return (GType) audio2video_type;
}

static inline GstAudio2VideoPrivate *
gst_audio2video_get_instance_private (GstAudio2Video * self)
{
  return (GstAudio2VideoPrivate*)(G_STRUCT_MEMBER_P (self, private_offset));
}

static void
gst_audio2video_class_init (GstAudio2VideoClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *element_class = (GstElementClass *) klass;

  if (private_offset != 0)
    g_type_class_adjust_private_offset (klass, &private_offset);

  parent_class = (GstBaseTransformClass*)g_type_class_peek_parent (klass);

  GST_DEBUG_CATEGORY_INIT (audio2video_debug,
      "baseaudio2video-libvisual", 0,
      "scope audio2video base class");

  gobject_class->dispose = gst_audio2video_dispose;

  element_class->change_state =
      GST_DEBUG_FUNCPTR (gst_audio2video_change_state);

  klass->decide_allocation = GST_DEBUG_FUNCPTR (default_decide_allocation);

  klass->get_latency = GST_DEBUG_FUNCPTR (gst_audio2video_get_latency);
}

static void
gst_audio2video_init (GstAudio2Video * scope,
    GstAudio2VideoClass * g_class)
{
  GstPadTemplate *pad_template;

  scope->priv = gst_audio2video_get_instance_private (scope);

  /* create the sink and src pads */
  pad_template =
      gst_element_class_get_pad_template (GST_ELEMENT_CLASS (g_class), "sink");
  g_return_if_fail (pad_template != NULL);
  scope->priv->sinkpad = gst_pad_new_from_template (pad_template, "sink");
  gst_pad_set_chain_function (scope->priv->sinkpad,
      GST_DEBUG_FUNCPTR (gst_audio2video_chain));
  gst_pad_set_event_function (scope->priv->sinkpad,
      GST_DEBUG_FUNCPTR (gst_audio2video_sink_event));
  gst_element_add_pad (GST_ELEMENT (scope), scope->priv->sinkpad);

  pad_template =
      gst_element_class_get_pad_template (GST_ELEMENT_CLASS (g_class), "src");
  g_return_if_fail (pad_template != NULL);
  scope->priv->srcpad = gst_pad_new_from_template (pad_template, "src");
  gst_pad_set_event_function (scope->priv->srcpad,
      GST_DEBUG_FUNCPTR (gst_audio2video_src_event));
  gst_pad_set_query_function (scope->priv->srcpad,
      GST_DEBUG_FUNCPTR (gst_audio2video_src_query));
  gst_element_add_pad (GST_ELEMENT (scope), scope->priv->srcpad);

  scope->priv->adapter = gst_adapter_new ();
  scope->priv->inbuf = gst_buffer_new ();

  /* reset the initial video state */
  gst_video_info_init (&scope->vinfo);
  scope->priv->frame_duration = GST_CLOCK_TIME_NONE;

  /* reset the initial state */
  gst_audio_info_init (&scope->ainfo);
  gst_video_info_init (&scope->vinfo);

  g_mutex_init (&scope->priv->config_lock);
}

static void
gst_audio2video_dispose (GObject * object)
{
  GstAudio2Video *scope = GST_AUDIO2VIDEO (object);

  if (scope->priv->adapter) {
    g_object_unref (scope->priv->adapter);
    scope->priv->adapter = NULL;
  }
  if (scope->priv->inbuf) {
    gst_buffer_unref (scope->priv->inbuf);
    scope->priv->inbuf = NULL;
  }
  if (scope->priv->tempbuf) {
    gst_buffer_unref (scope->priv->tempbuf);
    scope->priv->tempbuf = NULL;
  }
  if (scope->priv->config_lock.p) {
    g_mutex_clear (&scope->priv->config_lock);
    scope->priv->config_lock.p = NULL;
  }
  G_OBJECT_CLASS (parent_class)->dispose (object);
}

static void
gst_audio2video_reset (GstAudio2Video * scope)
{
  gst_adapter_clear (scope->priv->adapter);
  gst_segment_init (&scope->priv->segment, GST_FORMAT_UNDEFINED);

  GST_OBJECT_LOCK (scope);
  scope->priv->proportion = 1.0;
  scope->priv->earliest_time = -1;
  scope->priv->dropped = 0;
  scope->priv->processed = 0;
  GST_OBJECT_UNLOCK (scope);
}

static gboolean
gst_audio2video_sink_setcaps (GstAudio2Video * scope, GstCaps * caps)
{
  GstAudioInfo info;

  if (!gst_audio_info_from_caps (&info, caps))
    goto wrong_caps;

  scope->ainfo = info;

  GST_DEBUG_OBJECT (scope, "audio: channels %d, rate %d",
      GST_AUDIO_INFO_CHANNELS (&info), GST_AUDIO_INFO_RATE (&info));

  if (!gst_audio2video_src_negotiate (scope)) {
    goto not_negotiated;
  }

  return TRUE;

  /* Errors */
wrong_caps:
  {
    GST_WARNING_OBJECT (scope, "could not parse caps");
    return FALSE;
  }
not_negotiated:
  {
    GST_WARNING_OBJECT (scope, "failed to negotiate");
    return FALSE;
  }
}

static gboolean
gst_audio2video_src_setcaps (GstAudio2Video * scope, GstCaps * caps)
{
  GstVideoInfo info;
  GstAudio2VideoClass *klass;
  gboolean res;

  if (!gst_video_info_from_caps (&info, caps))
    goto wrong_caps;

  klass = GST_AUDIO2VIDEO_CLASS (G_OBJECT_GET_CLASS (scope));

  scope->vinfo = info;

  scope->priv->frame_duration = gst_util_uint64_scale_int (GST_SECOND,
      GST_VIDEO_INFO_FPS_D (&info), GST_VIDEO_INFO_FPS_N (&info));
  scope->priv->spf =
      gst_util_uint64_scale_int (GST_AUDIO_INFO_RATE (&scope->ainfo),
      GST_VIDEO_INFO_FPS_D (&info), GST_VIDEO_INFO_FPS_N (&info));
  scope->req_spf = scope->priv->spf;

  if (scope->priv->tempbuf) {
    gst_buffer_unref (scope->priv->tempbuf);
  }
  scope->priv->tempbuf = gst_buffer_new_wrapped (g_malloc0 (scope->vinfo.size),
      scope->vinfo.size);

  if (klass->setup && !klass->setup (scope))
    goto setup_failed;

  GST_DEBUG_OBJECT (scope, "video: dimension %dx%d, framerate %d/%d",
      GST_VIDEO_INFO_WIDTH (&info), GST_VIDEO_INFO_HEIGHT (&info),
      GST_VIDEO_INFO_FPS_N (&info), GST_VIDEO_INFO_FPS_D (&info));
  GST_DEBUG_OBJECT (scope, "blocks: spf %u, req_spf %u",
      scope->priv->spf, scope->req_spf);

  gst_pad_set_caps (scope->priv->srcpad, caps);

  /* find a pool for the negotiated caps now */
  res = gst_audio2video_do_bufferpool (scope, caps);
  gst_caps_unref (caps);

  return res;

  /* ERRORS */
wrong_caps:
  {
    gst_caps_unref (caps);
    GST_DEBUG_OBJECT (scope, "error parsing caps");
    return FALSE;
  }

setup_failed:
  {
    GST_WARNING_OBJECT (scope, "failed to set up");
    return FALSE;
  }
}

static gboolean
gst_audio2video_src_negotiate (GstAudio2Video * scope)
{
  GstCaps *othercaps, *target;
  GstStructure *structure;
  GstCaps *templ;
  gboolean ret;

  templ = gst_pad_get_pad_template_caps (scope->priv->srcpad);

  GST_DEBUG_OBJECT (scope, "performing negotiation");

  /* see what the peer can do */
  othercaps = gst_pad_peer_query_caps (scope->priv->srcpad, NULL);
  if (othercaps) {
    target = gst_caps_intersect (othercaps, templ);
    gst_caps_unref (othercaps);
    gst_caps_unref (templ);

    if (gst_caps_is_empty (target))
      goto no_format;

    target = gst_caps_truncate (target);
  } else {
    target = templ;
  }

  target = gst_caps_make_writable (target);
  structure = gst_caps_get_structure (target, 0);
  gst_structure_fixate_field_nearest_int (structure, "width", 320);
  gst_structure_fixate_field_nearest_int (structure, "height", 200);
  gst_structure_fixate_field_nearest_fraction (structure, "framerate", 25, 1);
  if (gst_structure_has_field (structure, "pixel-aspect-ratio"))
    gst_structure_fixate_field_nearest_fraction (structure,
        "pixel-aspect-ratio", 1, 1);

  target = gst_caps_fixate (target);

  GST_DEBUG_OBJECT (scope, "final caps are %" GST_PTR_FORMAT, target);

  ret = gst_audio2video_src_setcaps (scope, target);

  return ret;

no_format:
  {
    gst_caps_unref (target);
    return FALSE;
  }
}

/* takes ownership of the pool, allocator and query */
static gboolean
gst_audio2video_set_allocation (GstAudio2Video * scope,
    GstBufferPool * pool, GstAllocator * allocator,
    const GstAllocationParams * params, GstQuery * query)
{
  GstAllocator *oldalloc;
  GstBufferPool *oldpool;
  GstQuery *oldquery;
  GstAudio2VideoPrivate *priv = scope->priv;

  GST_OBJECT_LOCK (scope);
  oldpool = priv->pool;
  priv->pool = pool;
  priv->pool_active = FALSE;

  oldalloc = priv->allocator;
  priv->allocator = allocator;

  oldquery = priv->query;
  priv->query = query;

  if (params)
    priv->params = *params;
  else
    gst_allocation_params_init (&priv->params);
  GST_OBJECT_UNLOCK (scope);

  if (oldpool) {
    GST_DEBUG_OBJECT (scope, "deactivating old pool %p", oldpool);
    gst_buffer_pool_set_active (oldpool, FALSE);
    gst_object_unref (oldpool);
  }
  if (oldalloc) {
    gst_object_unref (oldalloc);
  }
  if (oldquery) {
    gst_query_unref (oldquery);
  }
  return TRUE;
}

static gboolean
gst_audio2video_do_bufferpool (GstAudio2Video * scope,
    GstCaps * outcaps)
{
  GstQuery *query;
  gboolean result = TRUE;
  GstBufferPool *pool = NULL;
  GstAudio2VideoClass *klass;
  GstAllocator *allocator;
  GstAllocationParams params;

  /* not passthrough, we need to allocate */
  /* find a pool for the negotiated caps now */
  GST_DEBUG_OBJECT (scope, "doing allocation query");
  query = gst_query_new_allocation (outcaps, TRUE);

  if (!gst_pad_peer_query (scope->priv->srcpad, query)) {
    /* not a problem, we use the query defaults */
    GST_DEBUG_OBJECT (scope, "allocation query failed");
  }

  klass = GST_AUDIO2VIDEO_GET_CLASS (scope);

  GST_DEBUG_OBJECT (scope, "calling decide_allocation");
  g_assert (klass->decide_allocation != NULL);
  result = klass->decide_allocation (scope, query);

  GST_DEBUG_OBJECT (scope, "ALLOCATION (%d) params: %" GST_PTR_FORMAT, result,
      query);

  if (!result)
    goto no_decide_allocation;

  /* we got configuration from our peer or the decide_allocation method,
   * parse them */
  if (gst_query_get_n_allocation_params (query) > 0) {
    gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
  } else {
    allocator = NULL;
    gst_allocation_params_init (&params);
  }

  if (gst_query_get_n_allocation_pools (query) > 0)
    gst_query_parse_nth_allocation_pool (query, 0, &pool, NULL, NULL, NULL);

  /* now store */
  result =
      gst_audio2video_set_allocation (scope, pool, allocator, &params,
      query);

  return result;

  /* Errors */
no_decide_allocation:
  {
    GST_WARNING_OBJECT (scope, "Subclass failed to decide allocation");
    gst_query_unref (query);

    return result;
  }
}

static gboolean
default_decide_allocation (GstAudio2Video * scope, GstQuery * query)
{
  GstCaps *outcaps;
  GstBufferPool *pool;
  guint size, min, max;
  GstAllocator *allocator;
  GstAllocationParams params;
  GstStructure *config;
  gboolean update_allocator;
  gboolean update_pool;

  gst_query_parse_allocation (query, &outcaps, NULL);

  /* we got configuration from our peer or the decide_allocation method,
   * parse them */
  if (gst_query_get_n_allocation_params (query) > 0) {
    /* try the allocator */
    gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
    update_allocator = TRUE;
  } else {
    allocator = NULL;
    gst_allocation_params_init (&params);
    update_allocator = FALSE;
  }

  if (gst_query_get_n_allocation_pools (query) > 0) {
    gst_query_parse_nth_allocation_pool (query, 0, &pool, &size, &min, &max);
    update_pool = TRUE;
  } else {
    pool = NULL;
    size = GST_VIDEO_INFO_SIZE (&scope->vinfo);
    min = max = 0;
    update_pool = FALSE;
  }

  if (pool == NULL) {
    /* we did not get a pool, make one ourselves then */
    pool = gst_video_buffer_pool_new ();
  }

  config = gst_buffer_pool_get_config (pool);
  gst_buffer_pool_config_set_params (config, outcaps, size, min, max);
  gst_buffer_pool_config_set_allocator (config, allocator, &params);
  gst_buffer_pool_config_add_option (config, GST_BUFFER_POOL_OPTION_VIDEO_META);
  gst_buffer_pool_set_config (pool, config);

  if (update_allocator)
    gst_query_set_nth_allocation_param (query, 0, allocator, &params);
  else
    gst_query_add_allocation_param (query, allocator, &params);

  if (allocator)
    gst_object_unref (allocator);

  if (update_pool)
    gst_query_set_nth_allocation_pool (query, 0, pool, size, min, max);
  else
    gst_query_add_allocation_pool (query, pool, size, min, max);

  if (pool)
    gst_object_unref (pool);

  return TRUE;
}

static GstFlowReturn
default_prepare_output_buffer (GstAudio2Video * scope, GstBuffer ** outbuf)
{
  GstAudio2VideoPrivate *priv;

  priv = scope->priv;

  g_assert (priv->pool != NULL);

  /* we can't reuse the input buffer */
  if (!priv->pool_active) {
    GST_DEBUG_OBJECT (scope, "setting pool %p active", priv->pool);
    if (!gst_buffer_pool_set_active (priv->pool, TRUE))
      goto activate_failed;
    priv->pool_active = TRUE;
  }
  GST_DEBUG_OBJECT (scope, "using pool alloc");

  return gst_buffer_pool_acquire_buffer (priv->pool, outbuf, NULL);

  /* ERRORS */
activate_failed:
  {
    GST_ELEMENT_ERROR (scope, RESOURCE, SETTINGS,
        ("failed to activate bufferpool"), ("failed to activate bufferpool"));
    return GST_FLOW_ERROR;
  }
}

static GstFlowReturn
gst_audio2video_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buffer)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstAudio2Video *scope;
  GstAudio2VideoClass *klass;
  GstBuffer *inbuf;
  guint64 dist, ts;
  guint avail, sbpf;
  gpointer adata;
  gint bpf, rate;

  scope = GST_AUDIO2VIDEO (parent);
  klass = GST_AUDIO2VIDEO_CLASS (G_OBJECT_GET_CLASS (scope));

  GST_LOG_OBJECT (scope, "chainfunc called");

  /* resync on DISCONT */
  if (GST_BUFFER_FLAG_IS_SET (buffer, GST_BUFFER_FLAG_DISCONT)) {
    gst_adapter_clear (scope->priv->adapter);
  }

  /* Make sure have an output format */
  if (gst_pad_check_reconfigure (scope->priv->srcpad)) {
    if (!gst_audio2video_src_negotiate (scope)) {
      gst_pad_mark_reconfigure (scope->priv->srcpad);
      goto not_negotiated;
    }
  }

  rate = GST_AUDIO_INFO_RATE (&scope->ainfo);
  bpf = GST_AUDIO_INFO_BPF (&scope->ainfo);

  if (bpf == 0) {
    ret = GST_FLOW_NOT_NEGOTIATED;
    goto beach;
  }

  gst_adapter_push (scope->priv->adapter, buffer);

  g_mutex_lock (&scope->priv->config_lock);

  /* this is what we want */
  sbpf = scope->req_spf * bpf;

  inbuf = scope->priv->inbuf;
  /* FIXME: the timestamp in the adapter would be different */
  gst_buffer_copy_into (inbuf, buffer, GST_BUFFER_COPY_METADATA, 0, -1);

  /* this is what we have */
  avail = gst_adapter_available (scope->priv->adapter);
  GST_LOG_OBJECT (scope, "avail: %u, bpf: %u", avail, sbpf);
  while (avail >= sbpf) {
    GstBuffer *outbuf;
    GstVideoFrame outframe;

    /* get timestamp of the current adapter content */
    ts = gst_adapter_prev_pts (scope->priv->adapter, &dist);
    if (GST_CLOCK_TIME_IS_VALID (ts)) {
      /* convert bytes to time */
      ts += gst_util_uint64_scale_int (dist, GST_SECOND, rate * bpf);
    }

    /* check for QoS, don't compute buffers that are known to be late */
    if (GST_CLOCK_TIME_IS_VALID (ts)) {
      GstClockTime earliest_time;
      gdouble proportion;
      gint64 qostime;

      qostime =
          gst_segment_to_running_time (&scope->priv->segment,
          GST_FORMAT_TIME, ts) + scope->priv->frame_duration;

      GST_OBJECT_LOCK (scope);
      earliest_time = scope->priv->earliest_time;
      proportion = scope->priv->proportion;
      GST_OBJECT_UNLOCK (scope);

      if (GST_CLOCK_TIME_IS_VALID (earliest_time) && qostime <= (gint64)earliest_time) {
        GstClockTime stream_time, jitter;
        GstMessage *qos_msg;

        GST_DEBUG_OBJECT (scope,
            "QoS: skip ts: %" GST_TIME_FORMAT ", earliest: %" GST_TIME_FORMAT,
            GST_TIME_ARGS (qostime), GST_TIME_ARGS (earliest_time));

        ++scope->priv->dropped;
        stream_time = gst_segment_to_stream_time (&scope->priv->segment,
            GST_FORMAT_TIME, ts);
        jitter = GST_CLOCK_DIFF (qostime, earliest_time);
        qos_msg = gst_message_new_qos (GST_OBJECT (scope), FALSE, qostime,
            stream_time, ts, GST_BUFFER_DURATION (buffer));
        gst_message_set_qos_values (qos_msg, jitter, proportion, 1000000);
        gst_message_set_qos_stats (qos_msg, GST_FORMAT_BUFFERS,
            scope->priv->processed, scope->priv->dropped);
        gst_element_post_message (GST_ELEMENT (scope), qos_msg);

        goto skip;
      }
    }

    ++scope->priv->processed;

    g_mutex_unlock (&scope->priv->config_lock);
    ret = default_prepare_output_buffer (scope, &outbuf);
    g_mutex_lock (&scope->priv->config_lock);
    /* recheck as the value could have changed */
    sbpf = scope->req_spf * bpf;

    /* no buffer allocated, we don't care why. */
    if (ret != GST_FLOW_OK)
      break;

    /* sync controlled properties */
    if (GST_CLOCK_TIME_IS_VALID (ts))
      gst_object_sync_values (GST_OBJECT (scope), ts);

    GST_BUFFER_PTS (outbuf) = ts;
    GST_BUFFER_DURATION (outbuf) = scope->priv->frame_duration;
    outframe.buffer=outbuf;

    /* this can fail as the data size we need could have changed */
    if (!(adata = (gpointer) gst_adapter_map (scope->priv->adapter, sbpf)))
      break;

    gst_buffer_replace_all_memory (inbuf,
        gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY, adata, sbpf, 0,
            sbpf, NULL, NULL));

    /* call class->render() vmethod */
    if (klass->render) {
      if (!klass->render (scope, inbuf, &outframe)) {
        ret = GST_FLOW_ERROR;
        g_mutex_unlock (&scope->priv->config_lock);
        goto beach;
      } 
    }
    gst_buffer_unref(outbuf);

  skip:
    /* recheck as the value could have changed */
    sbpf = scope->req_spf * bpf;
    GST_LOG_OBJECT (scope, "avail: %u, bpf: %u", avail, sbpf);
    /* we want to take less or more, depending on spf : req_spf */
    if (avail - sbpf >= sbpf) {
      gst_adapter_flush (scope->priv->adapter, sbpf);
      gst_adapter_unmap (scope->priv->adapter);
    } else if (avail >= sbpf) {
      /* just flush a bit and stop */
      gst_adapter_flush (scope->priv->adapter, sbpf);
      gst_adapter_unmap (scope->priv->adapter);
      break;
    }
    avail = gst_adapter_available (scope->priv->adapter);

    if (ret != GST_FLOW_OK)
      break;
  }

  g_mutex_unlock (&scope->priv->config_lock);

beach:
  return ret;

  /* ERRORS */
not_negotiated:
  {
    GST_DEBUG_OBJECT (scope, "Failed to renegotiate");
    return GST_FLOW_NOT_NEGOTIATED;
  }
}

static gboolean
gst_audio2video_src_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  gboolean res;
  GstAudio2Video *scope;

  scope = GST_AUDIO2VIDEO (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_QOS:
    {
      gdouble proportion;
      GstClockTimeDiff diff;
      GstClockTime timestamp;

      gst_event_parse_qos (event, NULL, &proportion, &diff, &timestamp);

      /* save stuff for the _chain() function */
      GST_OBJECT_LOCK (scope);
      scope->priv->proportion = proportion;
      if (diff >= 0)
        /* we're late, this is a good estimate for next displayable
         * frame (see part-qos.txt) */
        scope->priv->earliest_time = timestamp + 2 * diff +
            scope->priv->frame_duration;
      else
        scope->priv->earliest_time = timestamp + diff;
      GST_OBJECT_UNLOCK (scope);

      res = gst_pad_push_event (scope->priv->sinkpad, event);
      break;
    }
    case GST_EVENT_RECONFIGURE:
      /* don't forward */
      gst_event_unref (event);
      res = TRUE;
      break;
    default:
      res = gst_pad_event_default (pad, parent, event);
      break;
  }

  return res;
}

static gboolean
gst_audio2video_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  gboolean res;
  GstAudio2Video *scope;

  scope = GST_AUDIO2VIDEO (parent);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;

      gst_event_parse_caps (event, &caps);
      res = gst_audio2video_sink_setcaps (scope, caps);
      gst_event_unref (event);
      break;
    }
    case GST_EVENT_FLUSH_STOP:
      gst_audio2video_reset (scope);
      res = gst_pad_push_event (scope->priv->srcpad, event);
      break;
    case GST_EVENT_SEGMENT:
    {
      /* the newsegment values are used to clip the input samples
       * and to convert the incoming timestamps to running time so
       * we can do QoS */
      gst_event_copy_segment (event, &scope->priv->segment);

      res = gst_pad_push_event (scope->priv->srcpad, event);
      break;
    }
    default:
      res = gst_pad_event_default (pad, parent, event);
      break;
  }

  return res;
}

static gboolean
gst_audio2video_src_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  gboolean res = FALSE;
  GstAudio2Video *scope;

  scope = GST_AUDIO2VIDEO (parent);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_LATENCY:
    {
      /* We need to send the query upstream and add the returned latency to our
       * own */
      GstClockTime min_latency, max_latency;
      gboolean us_live;
      GstClockTime our_latency;
      guint max_samples;
      gint rate = GST_AUDIO_INFO_RATE (&scope->ainfo);

      if (rate == 0)
        break;

      if ((res = gst_pad_peer_query (scope->priv->sinkpad, query))) {
        gst_query_parse_latency (query, &us_live, &min_latency, &max_latency);

        GST_DEBUG_OBJECT (scope, "Peer latency: min %"
            GST_TIME_FORMAT " max %" GST_TIME_FORMAT,
            GST_TIME_ARGS (min_latency), GST_TIME_ARGS (max_latency));

        /* the max samples we must buffer buffer */
        max_samples = MAX (scope->req_spf, scope->priv->spf);
        our_latency = gst_util_uint64_scale_int (max_samples, GST_SECOND, rate);

        GST_DEBUG_OBJECT (scope, "Our latency: %" GST_TIME_FORMAT,
            GST_TIME_ARGS (our_latency));

        /* we add some latency but only if we need to buffer more than what
         * upstream gives us */
        min_latency += our_latency;
        if ((int)max_latency != -1)
          max_latency += our_latency;

        GST_DEBUG_OBJECT (scope, "Calculated total latency : min %"
            GST_TIME_FORMAT " max %" GST_TIME_FORMAT,
            GST_TIME_ARGS (min_latency), GST_TIME_ARGS (max_latency));

        gst_query_set_latency (query, TRUE, min_latency, max_latency);
      }
      break;
    }
    default:
      res = gst_pad_query_default (pad, parent, query);
      break;
  }

  return res;
}

static GstClockTime gst_audio2video_get_latency(GstAudio2Video * scope)
{
  return gst_util_uint64_scale_int(
    MAX(scope->req_spf, scope->priv->spf), GST_SECOND, 
    GST_AUDIO_INFO_RATE (&scope->ainfo));
}

static GstStateChangeReturn
gst_audio2video_change_state (GstElement * element,
    GstStateChange transition)
{
  GstStateChangeReturn ret;
  GstAudio2Video *scope;

  scope = GST_AUDIO2VIDEO (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_audio2video_reset (scope);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_audio2video_set_allocation (scope, NULL, NULL, NULL, NULL);
      break;
    case GST_STATE_CHANGE_READY_TO_NULL:
      break;
    default:
      break;
  }

  return ret;
}
