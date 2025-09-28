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

#include "deepstream_asr_app.h"
#include <string.h>

static gboolean
bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  StreamCtx *sctx = (StreamCtx *) data;
  switch (GST_MESSAGE_TYPE (message)) {
    case GST_MESSAGE_INFO:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_info (message, &error, &debuginfo);
      g_printerr ("INFO from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_WARNING:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_warning (message, &error, &debuginfo);
      g_printerr ("WARNING from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      g_error_free (error);
      g_free (debuginfo);
      break;
    }
    case GST_MESSAGE_ERROR:{
      GError *error = NULL;
      gchar *debuginfo = NULL;
      gst_message_parse_error (message, &error, &debuginfo);
      g_printerr ("ERROR from %s: %s\n",
          GST_OBJECT_NAME (message->src), error->message);
      if (debuginfo) {
        g_printerr ("Debug info: %s\n", debuginfo);
      }
      break;
    }
    case GST_MESSAGE_STATE_CHANGED:{
      break;
    }
    case GST_MESSAGE_EOS:{
      /*
       * In normal scenario, this would use g_main_loop_quit() to exit the
       * loop and release the resources. Since this application might be
       * running multiple pipelines through configuration files, it should wait
       * till all pipelines are done.
       */
      sctx->eos_received = TRUE;
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps (decoder_src_pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;

  /* Need to check if the pad created by the decodebin is for audio */
  if (!strncmp (name, "audio", 5)) {
    /* Get the source bin ghost pad */
    GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "asrc");

    if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
            decoder_src_pad)) {
      g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
    }
    gst_object_unref (bin_ghost_pad);

    gst_element_sync_state_with_parent (source_bin);
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, StreamCtx * sctx)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", sctx->uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);

  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("asrc",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

static GstPadProbeReturn
decoder_src_pad_event_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT (info);
  StreamCtx *sctx = (StreamCtx *) u_data;
  GstCaps *caps;
  GstStructure *s;
  int num_channels = 0;
  int sampling_rate = 0;

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    gst_event_parse_caps (event, &caps);
    s = gst_caps_get_structure (caps, 0);
    gst_structure_get_int (s, "channels", &num_channels);
    gst_structure_get_int (s, "rate", &sampling_rate);
    printf ("num_channels = %d sampling rate = %d\n", num_channels,
        sampling_rate);

    if (num_channels != 1 || sampling_rate != 16000) {
      g_printerr
          ("Error !!! Only single channel with 16k sampling rate is supported...exiting\n");
      exit (-1);
    }
  }
  sctx->has_audio = TRUE;
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
asr_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  StreamCtx *sctx = (StreamCtx *) u_data;

  /* Write text output to file TODO */
  GstBuffer *buf = (GstBuffer *) info->data;
  char *text_data = NULL;
  char newline = '\n';
  GstMapInfo inmap = GST_MAP_INFO_INIT;

  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    g_printerr ("Unable to map info from buffer\n");
    return GST_PAD_PROBE_DROP;
  }
  text_data = (char *)inmap.data;
  fwrite (text_data, strlen (text_data), 1, sctx->FP_asr);
  fwrite (&newline, 1, 1, sctx->FP_asr);
  return GST_PAD_PROBE_OK;
}

static int
create_asr_pipeline (AppCtx * appctx, int stream_num, StreamCtx * sctx,
    GstElement * pipeline, GstElement * source_bin)
{
  /* Gst elements */
  GstElement *tee = NULL;
  GstElement *audio_resampler = NULL, *asr = NULL, *displaysink = NULL;
  GstElement *audio_sink = NULL;

  /* Gst pads */
  GstPad *link_sinkpad = NULL, *decoder_srcpad = NULL;
  GstPad *resampler_sinkpad = NULL, *asr_srcpad = NULL;
  GstPad *tee_resampler_srcpad = NULL, *tee_renderer_srcpad = NULL;

  /* create tee element */
  tee = gst_element_factory_make ("tee", "tee");

  /* Create audio_resampler element */
  audio_resampler =
      gst_element_factory_make ("audioresample", "audioresampler");

  /* Create ASR element */
  asr = gst_element_factory_make ("nvdsasr", "nvasr");

  g_object_set (G_OBJECT (asr), "config-file", "riva_asr_grpc_jasper_conf.yml", NULL);
  g_object_set (G_OBJECT (asr), "customlib-name", "libnvds_riva_asr_grpc.so", NULL);
  g_object_set (G_OBJECT (asr), "create-speech-ctx-func", "create_riva_asr_grpc_ctx", NULL);

   /* create audio renderer component to play input audio */
  if (sctx->audio_config.enable_playback) {
    audio_sink = gst_element_factory_make ("autoaudiosink", "audio-renderer");
  } else {
    audio_sink = gst_element_factory_make ("fakesink", "audio_fakesink");
  }

  displaysink = gst_element_factory_make("fakesink", "display_fakesink");

  if ( !tee || !audio_resampler || !asr || !displaysink || !audio_sink) {
    g_printerr
        ("Failed to create one of the components in audio pipeline. Exiting.\n");
    return -1;
  }

  gst_bin_add_many (GST_BIN (pipeline), tee,
      audio_resampler, asr, displaysink, audio_sink, NULL);

  if (sctx->audio_config.enable_playback)
  {
    g_object_set (G_OBJECT (audio_sink), "async-handling", TRUE, NULL);
  }
  else
  {
    g_object_set (G_OBJECT (audio_sink), "async", FALSE, NULL);
  }

  g_object_set(G_OBJECT(audio_sink), "sync", sctx->audio_config.sync, NULL);

  g_object_set (G_OBJECT (displaysink), "sync", sctx->audio_config.sync, NULL);
  g_object_set(G_OBJECT(displaysink), "async", FALSE, NULL);

  /* linking process */
  link_sinkpad = gst_element_get_static_pad (tee, "sink");
  if (!link_sinkpad) {
    g_printerr ("audio_resampler sink pad failed. \n");
    return -1;
  }

  /* Get src pad of source bin */
  decoder_srcpad = gst_element_get_static_pad (source_bin, "asrc");
  if (!decoder_srcpad) {
    g_printerr ("Failed to get src pad of source bin\n");
    return -1;
  }

  /* link source bin source pad and tee sink pad */
  if (gst_pad_link (decoder_srcpad, link_sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link audio source bin to tee\n");
    return -1;
  }

  gst_object_unref (link_sinkpad);

 /* P1: decoder -> tee -> audio renderer */
  tee_renderer_srcpad = gst_element_request_pad_simple (tee, "src_%u");
  GstPad *renderer_pad = gst_element_get_static_pad (audio_sink, "sink");
  gst_pad_link (tee_renderer_srcpad, renderer_pad);


  /* P2: decoder -> tee -> resampler -> asr -> fakesink */
  tee_resampler_srcpad = gst_element_request_pad_simple (tee, "src_%u");
  resampler_sinkpad = gst_element_get_static_pad (audio_resampler, "sink");
  if (!resampler_sinkpad) {
    g_printerr ("audio_resampler sink pad failed. \n");
    return -1;
  }
  gst_pad_link (tee_resampler_srcpad, resampler_sinkpad);

  if (!gst_element_link_many (audio_resampler, asr, NULL)) {
    g_printerr ("Elements could not be linked. \n");
    return -1;
  }

  asr_srcpad = gst_element_get_static_pad (asr, "src");
  if (!asr_srcpad) {
    g_printerr ("Can not get asr_src pad\n");
    return -1;
  }
  /* get asr static pad to link to textoverlay and to write ASR output
     in the probe function */
  gst_element_link_many(asr, displaysink, NULL);

  if (sctx->audio_config.asr_output_file_name == NULL) {
    g_printerr ("In config file ASR output text file is not provided for stream %d \
     \n",
        stream_num);
    exit (-1);
  }

  sctx->FP_asr = fopen (sctx->audio_config.asr_output_file_name, "w");
  if (sctx->FP_asr == NULL) {
    g_printerr ("Can not open ASR output text file\n");
    exit (-1);
  }

  /* Lets add probe to get informed if the stream has audio data */
  if (decoder_srcpad) {
    gst_pad_add_probe (decoder_srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        decoder_src_pad_event_probe, sctx, NULL);
  }
  gst_object_unref (decoder_srcpad);

  gst_object_unref (resampler_sinkpad);
  /* Lets add probe to get ASR output */
  if (asr_srcpad) {
    gst_pad_add_probe (asr_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
        asr_src_pad_buffer_probe, sctx, NULL);
    gst_object_unref (asr_srcpad);
    asr_srcpad = NULL;
  }

  return 0;
}

int
create_pipeline (AppCtx * appctx, int stream_num, StreamCtx * sctx)
{
  GstElement *pipeline = NULL, *source_bin = NULL;
  GstBus *bus = NULL;
  int ret = 0;

  sctx->stream_id = stream_num;
  /* Create Pipeline element */
  pipeline = gst_pipeline_new ("ASR-pipeline");
  sctx->asr_pipeline = pipeline;

  /* create source bin */
  source_bin = create_source_bin (stream_num, sctx);

  if (!pipeline || !source_bin) {
    g_printerr ("Failed to create pipeline or source bin. Exiting.\n");
    return -1;
  }

  /* Add source_bin to pipeline */
  gst_bin_add (GST_BIN (pipeline), source_bin);

  /* create audio pipeline */
  ret = create_asr_pipeline (appctx, stream_num, sctx, pipeline, source_bin);

  if (ret != 0) {
    return 1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  sctx->bus_id = gst_bus_add_watch (bus, bus_callback, sctx);
  gst_object_unref (bus);

  return ret;
}

int
destroy_pipeline (StreamCtx * sctx)
{
  gst_element_set_state (sctx->asr_pipeline, GST_STATE_NULL);
  if (sctx->uri) {
    g_free (sctx->uri);
    sctx->uri = NULL;
  }
  if (sctx->FP_asr) {
    fclose (sctx->FP_asr);
    sctx->FP_asr = NULL;
  }
  return 0;
}

int
start_pipeline (int stream_num, StreamCtx * sctx)
{
  /* Set Pipeline in PLAYING state */
  if (gst_element_set_state (sctx->asr_pipeline,
          GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    return -1;
  }
  return 0;
}
