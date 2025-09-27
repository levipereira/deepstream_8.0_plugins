/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */


#include "deepstream_asr_tts_app.h"
#include <string.h>

extern GMainLoop *loop;

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
      sctx->eos_received = TRUE;
      g_main_loop_quit (loop);
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
  uri_decode_bin = gst_element_factory_make ("uridecodebin3", "uri-decode-bin");

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
   * for the audio decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the audio decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the audio decoder
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
  int channel;

  if (GST_EVENT_TYPE (event) == GST_EVENT_CAPS) {
    GValue v = G_VALUE_INIT;
    GValue v2 = G_VALUE_INIT;
    GValue v3 = G_VALUE_INIT;

    gst_event_parse_caps (event, &caps);
    s = gst_caps_get_structure (caps, 0);
    gst_structure_get_int (s, "channels", &num_channels);
    gst_structure_get_int (s, "rate", &sampling_rate);
    g_print ("Num_channels = %d sampling rate = %d\n", num_channels,
        sampling_rate);

    /* Configure downmix from input number of channels to mono audio */
    g_object_set(G_OBJECT(sctx->input_downmixer), "in-channels", num_channels, NULL);

    g_value_init (&v2, GST_TYPE_ARRAY);
    g_value_init (&v, GST_TYPE_ARRAY);

    for (channel = 0; channel < num_channels; channel++) {
      g_value_init (&v3, G_TYPE_DOUBLE);
      g_value_set_double (&v3, (double)1 / num_channels);
      gst_value_array_append_value (&v2, &v3);
      g_value_unset (&v3);
    }

    gst_value_array_append_value (&v, &v2);
    g_value_unset (&v2);
    g_object_set_property (G_OBJECT (sctx->input_downmixer), "matrix", &v);
    g_value_unset (&v);

  }
  sctx->has_audio = TRUE;
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
asr_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  StreamCtx *sctx = (StreamCtx *) u_data;

  /* Write ASR output to file */
  GstBuffer *buf = (GstBuffer *) info->data;
  char *text_data = NULL;
  char newline = '\n';
  GstMapInfo inmap = GST_MAP_INFO_INIT;

  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    g_printerr ("Unable to map info from buffer\n");
    return GST_FLOW_ERROR;
  }
  text_data = (char *)inmap.data;
  fwrite (text_data, strlen (text_data), 1, sctx->FP_asr);
  fwrite (&newline, 1, 1, sctx->FP_asr);
  return GST_PAD_PROBE_OK;
}

/* With Gstreamer v1.16 (Ubuntu 20.04), proxy sink elements are not sending
 * EOS message to pipeline bus. Adding a probe for forwarding EOS, as a
 * workaround */
static GstPadProbeReturn
proxy_sink_pad_event_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT (info);

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    GstElement *component = GST_ELEMENT_PARENT (pad);
    gst_element_post_message (component, gst_message_new_eos( GST_OBJECT (component)));
  }
  return GST_PAD_PROBE_OK;
}


static int
create_asr_pipeline (AppCtx * appctx, int stream_num, StreamCtx * sctx,
    GstElement * pipeline, GstElement * source_bin,
    GstElement **p_proxy_audio_sink,
    guint enable_playback)
{
  /* Gst elements */
  GstElement *input_audio_convert = NULL;
  GstElement *input_downmixer = NULL;
  GstElement *audio_resampler = NULL;
  GstElement *audio_sink = NULL;
  GstElement *asr = NULL;
  GstElement *tts = NULL;
  GstElement *out_resampler = NULL;
  GstElement *audio_queue = NULL;

  /* Gst pads */
  GstPad *decoder_srcpad = NULL;
  GstPad *asr_srcpad = NULL;

  GstCaps *caps;

  GValue v = G_VALUE_INIT;
  GValue v2 = G_VALUE_INIT;
  GValue v3 = G_VALUE_INIT;

  /* Create audio converter for input non-interleaved to interleaved conversion */
  input_audio_convert = gst_element_factory_make ("audioconvert", "input_audio_convert");

  /* Create downmixer to convert input to single channel */
  input_downmixer = gst_element_factory_make ("audiomixmatrix", "input_downmixer");
  sctx->input_downmixer = input_downmixer;

  /* Create audio_resampler element */
  audio_resampler = gst_element_factory_make ("audioresample", "audio_resampler");

  /* Create ASR element */
  asr = gst_element_factory_make ("nvdsasr", "nvasr");

  /* Create TTS element */
  tts = gst_element_factory_make ("nvds_text_to_speech", "nvtts");


  if (!input_audio_convert || !input_downmixer || !audio_resampler || !asr || !tts
       ) {
    g_printerr
        ("Failed to create one of the components in audio pipeline. Exiting.\n");
    return -1;
  }

  gst_bin_add_many (GST_BIN (pipeline), input_audio_convert, input_downmixer,
      audio_resampler, asr, tts, NULL);

  /* Create audio renderer components if playback is enabled  */
  if (enable_playback) {
    /* Create queue for audio path to renderer */
    audio_queue = gst_element_factory_make ("queue2", "audio_queue");
    /* Create resampler to convert to 44.1 kHz for RTSP output */
    out_resampler = gst_element_factory_make ("audioresample", "out_resampler");
    *p_proxy_audio_sink = gst_element_factory_make ("proxysink", "proxy_audio_sink");

    if (!audio_queue || !out_resampler ||  !*p_proxy_audio_sink) {
        g_printerr ("Failed to create one of the components in audio pipeline. \
                Exiting.\n");
        return -1;
    }


    gst_bin_add_many(GST_BIN (pipeline), audio_queue, out_resampler,
        *p_proxy_audio_sink, NULL);
  } else {
    audio_sink = gst_element_factory_make ("fakesink", "fake_sink");
    if (!audio_sink) {
        g_printerr ("Failed to create one of the components in audio pipeline. \
                Exiting.\n");
        return -1;
    }
    gst_bin_add_many(GST_BIN (pipeline), audio_sink, NULL);
  }

  /* set properties on elements */

  g_object_set(G_OBJECT(input_downmixer), "out-channels", 1, NULL);
  g_object_set(G_OBJECT(input_downmixer), "channel-mask", 1, NULL);
  /* Default mono input */
  g_object_set(G_OBJECT(input_downmixer), "in-channels", 1, NULL);

  g_value_init (&v2, GST_TYPE_ARRAY);
  g_value_init (&v3, G_TYPE_DOUBLE);
  g_value_init (&v, GST_TYPE_ARRAY);

  g_value_set_double (&v3, (double)1);
  gst_value_array_append_value (&v2, &v3);
  g_value_unset (&v3);

  gst_value_array_append_value (&v, &v2);
  g_value_unset (&v2);
  g_object_set_property (G_OBJECT (input_downmixer), "matrix", &v);
  g_value_unset (&v);

  g_object_set (G_OBJECT (asr), "config-file", "riva_asr_grpc_conf.yml", NULL);
  g_object_set (G_OBJECT (asr), "customlib-name", "libnvds_riva_asr_grpc.so", NULL);
  g_object_set (G_OBJECT (asr), "create-speech-ctx-func", "create_riva_asr_grpc_ctx", NULL);

  g_object_set (G_OBJECT (tts), "config-file", "riva_tts_conf.yml", NULL);

  if (enable_playback)
  {
    g_object_set(G_OBJECT(audio_queue), "max-size-buffers", 1, NULL);
  }
  else
  {
    g_object_set (G_OBJECT (audio_sink), "async", FALSE, NULL);
  }


  /* linking process */

  /* Link source bin and input_audio_convert */
  if (!gst_element_link_pads (source_bin, "asrc", input_audio_convert, "sink")) {
    g_printerr ("Failed to link audio source bin to input_audio_convert\n");
    return -1;
  }

  if (!gst_element_link_many (input_audio_convert, input_downmixer,
              audio_resampler, NULL)) {
    g_printerr ("Elements could not be linked. \n");
    return -1;
  }

  /* Resample to 16 kHz signal required by ASR */
  caps = gst_caps_new_simple("audio/x-raw",
                             "format", G_TYPE_STRING, GST_AUDIO_NE(S16),
                             "rate", G_TYPE_INT, 16000,
                             "channels", G_TYPE_INT, 1,
                             NULL);
  if (!caps) {
    g_printerr ("audio_resampler caps failed. \n");
    return -1;
  }

  if (!gst_element_link_filtered(audio_resampler, asr, caps)) {
    g_printerr("Falied to link audio_resampler and asr. \n");
    return -1;
  }
  gst_caps_unref (caps);

  gst_element_link_many(asr, tts, NULL);

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


  if (enable_playback) {

    /* tts -> audio sink */

    if (!gst_element_link_many (tts, audio_queue, out_resampler, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }


    /* Resample to 48 kHz signal required by OPUS encoder for RTSP output */
    GstCaps* out_caps = gst_caps_new_simple("audio/x-raw",
        "format", G_TYPE_STRING, GST_AUDIO_NE(S16),
        "rate", G_TYPE_INT, 48000,
        "channels", G_TYPE_INT, 1,
        NULL);
    if (!out_caps) {
      g_printerr ("Creating out_resampler caps failed. \n");
      return -1;
    }

    if (!gst_element_link_filtered(out_resampler, *p_proxy_audio_sink, out_caps)) {
      g_printerr("Falied to link out_resampler and sink. \n");
      return -1;
    }
    gst_caps_unref (out_caps);

  } else {
    if (!gst_element_link_many (tts, audio_sink, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }
  }

  /* Lets add probe to get informed if the stream has audio data */
  decoder_srcpad = gst_element_get_static_pad (source_bin, "asrc");
  if (!decoder_srcpad) {
    g_printerr ("Failed to get src pad of source bin\n");
    return -1;
  }

  gst_pad_add_probe (decoder_srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
      decoder_src_pad_event_probe, sctx, NULL);
  gst_object_unref (decoder_srcpad);

  /* Lets add probe to get ASR output */
  asr_srcpad = gst_element_get_static_pad (asr, "src");
  if (!asr_srcpad) {
    g_printerr ("Can not get asr_src pad\n");
    return -1;
  }

  gst_pad_add_probe (asr_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
          asr_src_pad_buffer_probe, sctx, NULL);
  gst_object_unref (asr_srcpad);
  asr_srcpad = NULL;

  /* With Gstreamer v1.16 (Ubuntu 20.04), proxy sink elements are not sending
   * EOS message to pipeline bus. Attaching a probe for forwarding EOS, as a
   * workaround */
  if (enable_playback) {
    ATTACH_STATIC_PAD_PROBE(*p_proxy_audio_sink, "sink",
            GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
            proxy_sink_pad_event_probe, sctx);
  }

  return 0;
}

int
create_pipeline (AppCtx * appctx, int stream_num, StreamCtx * sctx,
    GstElement **p_proxy_audio_sink)
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
  ret = create_asr_pipeline (appctx, stream_num, sctx, pipeline, source_bin,
      p_proxy_audio_sink, appctx->enable_playback);

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

static gboolean
renderer_bus_callback (GstBus * bus, GstMessage * message, gpointer data)
{
  AppCtx *appctx = (AppCtx *) data;
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
      g_main_loop_quit (loop);
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
      appctx->eos_received = TRUE;
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int
create_renderer_pipeline (AppCtx * appctx)
{
  int ret = 0;
  unsigned int i = 0;

  /* Create Renderer Pipeline */
  GstElement *pipeline = NULL, *audio_mixer = NULL;
  GstElement *audio_sink= NULL;
  GstElement *audio_convert = NULL;
  GstElement *file_out = NULL;
  GstElement *audio_enc = NULL;

  GstElement *out_muxer = NULL;
  GstElement *muxer_aqueue = NULL;
  GstElement *out_aqueue = NULL;
  GstElement *rtsp_out_bin = NULL;
  guint enc_type = 0;
  guint rtsp_port_num = 8554;
  guint enable_playback = appctx->enable_playback;

  pipeline = gst_pipeline_new ("Renderer-pipeline");
  appctx->renderer_pipeline = pipeline;

  if ((enable_playback == 2) &&
      (appctx->playback_output_file_name == NULL)) {
    g_printerr("Playback output file name not available. \n");
    exit (-1);
  }

  if (enable_playback == 0) {
    g_printerr("Renderer pipeline not required. \n");
    return 0;
  }

  /* Create audio mixer component to mix audio sources */
  audio_mixer = gst_element_factory_make ("audiomixer", "audio_mixer");
  /* Create queue for audio output */
  out_aqueue = gst_element_factory_make ("queue2", "out_audio_queue");

  if (!pipeline || !audio_mixer || !out_aqueue) {
    g_printerr ("Failed to create one of the components in renderer pipeline. \
		 Exiting.\n");
    return -1;
  }

  gst_bin_add_many (GST_BIN (pipeline), audio_mixer, out_aqueue, NULL);

  if (enable_playback == 1) {
    /* Render using autoaudiosink */
    audio_sink = gst_element_factory_make ("autoaudiosink", "audio_renderer");

    if (!audio_sink) {
      g_printerr ("Failed to create renderer elements. Exiting.\n");
      return -1;
    }
    gst_bin_add_many (GST_BIN (pipeline), audio_sink, NULL);

  } else if (enable_playback == 4) {
    /* Render using pulsesink */
    audio_sink = gst_element_factory_make ("pulsesink", "audio_renderer");

    if (!audio_sink) {
      g_printerr ("Failed to create renderer elements. Exiting.\n");
      return -1;
    }
    gst_bin_add_many (GST_BIN (pipeline), audio_sink, NULL);

  } else if (enable_playback == 3) {
    /* Create RTSP output bin */
    rtsp_out_bin = gst_element_factory_make ("nvrtspoutsinkbin", "nvrtsp-renderer");

    if (!rtsp_out_bin) {
      g_printerr ("Failed to create RTSP output elements. Exiting.\n");
      return -1;
    }

    g_object_set (G_OBJECT (rtsp_out_bin), "sync", TRUE, NULL);
    g_object_set (G_OBJECT (rtsp_out_bin), "bitrate", 768000, NULL);
    g_object_set (G_OBJECT (rtsp_out_bin), "rtsp-port", rtsp_port_num, NULL);
    g_object_set (G_OBJECT (rtsp_out_bin), "enc-type", enc_type, NULL);

    gst_bin_add_many (GST_BIN (pipeline), rtsp_out_bin, NULL);

  } else if (enable_playback == 2) {

    /* Create file write components for encoded file output */
    audio_convert = gst_element_factory_make ("audioconvert", "audio_convert");
    file_out = gst_element_factory_make ("filesink", "file_sink");

    /* create audio encoder to encode audio data */
    audio_enc = gst_element_factory_make ("vorbisenc", "audio encoder");

    /* create audio muxer to mux audio data */
    if (!audio_convert || !audio_enc)  {
      g_printerr("Failed to create one of the components in pipeline. Exiting.\n");
      return -1;
    }

    gst_bin_add_many (GST_BIN (pipeline), audio_convert, audio_enc, NULL);


    out_muxer = gst_element_factory_make("matroskamux", "muxer");
    muxer_aqueue = gst_element_factory_make ("queue", "muxer_audio_queue");

    if (!out_muxer || !muxer_aqueue || !file_out)  {
      g_printerr("Failed to create one of the components in pipeline. Exiting.\n");
      return -1;
    }

    gst_bin_add_many (GST_BIN (pipeline), muxer_aqueue,
        out_muxer, file_out, NULL);
  }

  /* Set the element properties */

  if ((enable_playback == 1) || (enable_playback == 4))
  {
    g_object_set (G_OBJECT (out_aqueue), "max-size-buffers", 2560, NULL);
    g_object_set (G_OBJECT (out_aqueue), "max-size-time", 4000000000, NULL);
    if (enable_playback == 1) {
      g_object_set (G_OBJECT (audio_sink), "async-handling", TRUE, NULL);
    } else {
      /* For smoother playback in case of network delays,
       * add a two second buffer for pulsesink */
      g_object_set (G_OBJECT (audio_sink), "buffer-time", 2000000000, NULL);
    }
    g_object_set (G_OBJECT (audio_sink), "sync", appctx->sync, NULL);
  }
  else if (enable_playback == 2)
  {
    g_object_set (G_OBJECT (muxer_aqueue), "max-size-buffers", 10, NULL);

    g_object_set (G_OBJECT (file_out), "location",
        appctx->playback_output_file_name, NULL);
    g_object_set (G_OBJECT (file_out), "async", FALSE, NULL);
    g_object_set (G_OBJECT (file_out), "sync", TRUE, NULL);
  } else if (enable_playback == 3) {
    g_object_set (G_OBJECT (out_aqueue), "max-size-buffers", 1280, NULL);
    g_object_set (G_OBJECT (out_aqueue), "max-size-time", 16000000000, NULL);
  }

  /* Link the renderer pipeline with the ASR pipelines */
  for (i = 0; i < appctx->num_sources; i++) {
    gchar proxy_src_name[32] = { };
    g_snprintf (proxy_src_name, 31, "proxy_audio_src-%02d", i);

    appctx->proxy_audio_sources[i] =
      gst_element_factory_make ("proxysrc", proxy_src_name);

    if (!appctx->proxy_audio_sources[i]) {
      g_printerr ("Failed to create proxy source. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), appctx->proxy_audio_sources[i]);

    g_object_set (G_OBJECT (appctx->proxy_audio_sources[i]), "proxysink",
        appctx->proxy_audio_sinks[i], NULL);

  }

  for (i = 0; i < appctx->num_sources; i++) {

    /* Link proxy audio sources to audio mixer */
    if (!gst_element_link_many (appctx->proxy_audio_sources[i], audio_mixer, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }

  }

  /* audio mixer -> queue -> audio rendering */
  if (!gst_element_link_many (audio_mixer, out_aqueue, NULL)) {
    g_printerr ("Elements could not be linked. \n");
    return -1;
  }

  if (enable_playback == 2) {

    /* queue -> audio convert -> audio encode */
    if (!gst_element_link_many (out_aqueue, audio_convert, audio_enc, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }

    /* audio encode -> muxer queue -> output muxer */
    if (!gst_element_link_many (audio_enc, muxer_aqueue, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }
    LINK_STATIC_PAD_REQUEST_PAD(muxer_aqueue, "src", out_muxer, "audio_%u");

    if (!gst_element_link_many (out_muxer, file_out, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }

  } else if ((enable_playback == 1) || (enable_playback == 4))  {

    /* queue -> audio out*/
    if (!gst_element_link_many (out_aqueue, audio_sink, NULL)) {
      g_printerr ("Elements could not be linked. \n");
      return -1;
    }
  } else if (enable_playback == 3) {
    /* queue -> RTSP out audio */
    if (!gst_element_link_many (out_aqueue, rtsp_out_bin, NULL)) {
      g_printerr("Audio RTSP chain could not be linked. Exiting.\n");
      return -1;
    }

  }


  /* Add a message handler */
  GstBus *bus = NULL;
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  appctx->bus_id = gst_bus_add_watch (bus, renderer_bus_callback, appctx);
  gst_object_unref (bus);

  return ret;
}
