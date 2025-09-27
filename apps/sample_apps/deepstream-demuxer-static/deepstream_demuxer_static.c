/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvevent.h"

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

GstElement *pipeline = NULL, *nvmultiurisrcbin = NULL, *nvstreamdemux=NULL, *pgie = NULL;
static guint num_src_pad = 0;
static guint num_src_pad_used = 0;
static guint max_batch_size = 0;
GMainLoop *loop = NULL;
GQueue *ended_streams_queue = NULL;
struct cudaDeviceProp prop;
GMutex pipeline_mutex;

typedef struct {
    GstElement *input_selector;
    guint num_src_pad;
} ProbeData;

/**
 * This function is called when a pad deletion event is detected on a demuxer source pad.
 * It handles the stream end event by:
 * 1. Switching the input selector to a test pattern source
 * 2. Adding the pad number to a queue of ended streams
 * 3. Processing any previously ended streams
 */
static GstPadProbeReturn
demux_src_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    ProbeData *probe_data = (ProbeData *)user_data;
    GstElement * input_selector = probe_data->input_selector;
    guint num_src_pad = probe_data->num_src_pad;

    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
    {
        /* Process all ended streams in the queue while holding the mutex */
        g_mutex_lock(&pipeline_mutex);
        while (!g_queue_is_empty(ended_streams_queue)) {
            guint ended_pad_num = GPOINTER_TO_UINT(g_queue_peek_head(ended_streams_queue));
            gchar input_selector_name[26];
            snprintf(input_selector_name, sizeof(input_selector_name), "input_selector_%u", ended_pad_num);
            GstElement *current_input_selector = gst_bin_get_by_name(GST_BIN(pipeline), input_selector_name);
            if (current_input_selector) {
                GstPad *sink_pad = gst_element_get_static_pad(current_input_selector, "sink_1");
                if (sink_pad) {
                    g_object_set(current_input_selector, "active-pad", sink_pad, NULL);
                    gst_object_unref(sink_pad);
                }
                gst_object_unref(current_input_selector);
                /* Remove the processed pad number from the queue */
                g_queue_pop_head(ended_streams_queue);
            }
        }

        /* Set the current input selector to show videotestsrc */
        GstPad *sink_pad = gst_element_get_static_pad(input_selector, "sink_0");
        if (sink_pad) {
            g_object_set(input_selector, "active-pad", sink_pad, NULL);
            g_queue_push_tail(ended_streams_queue, GUINT_TO_POINTER(num_src_pad));
            gst_object_unref(sink_pad);
        }

        g_mutex_unlock(&pipeline_mutex);
    }
    return GST_PAD_PROBE_OK;
}


/**
 * Creates a new processing branch for a stream from nvstreamdemux.
 *
 * This function sets up a pipeline branch with the following structure:
 * nvstreamdemux_src_pad -> queue -> nvdsosd -> input_selector (sink_1)
 *                                               /
 * videotestsrc -> switch_queue --------------> / (sink_0)
 *                                             /
 *                           input_selector ---> sink
 *
 * It's called for each new source stream to create a dedicated
 * processing path with on-screen display and fallback pattern.
 *
 */
static int
create_branch(void)
{
    GstElement *queue=NULL, *nvosd=NULL, *sink=NULL, *input_selector=NULL,*videotestsrc=NULL,*switch_queue=NULL;
    GstPad *src_pad=NULL, *sink_pad=NULL;
    gchar q_name[16], osd_name[16], sink_name[16], pad_name[16], input_selector_name[26], videotestsrc_name[16], switch_queue_name[16];
    snprintf (q_name, sizeof(q_name), "queue_%u", num_src_pad);
    snprintf (osd_name, sizeof(osd_name), "nvosd_%u", num_src_pad);
    snprintf (sink_name, sizeof(sink_name), "sink_%u", num_src_pad);
    snprintf (pad_name, sizeof(pad_name), "src_%u", num_src_pad);
    snprintf (input_selector_name, sizeof(input_selector_name), "input_selector_%u", num_src_pad);
    snprintf (videotestsrc_name, sizeof(videotestsrc_name), "videotestsrc_%u", num_src_pad);
    snprintf (switch_queue_name, sizeof(switch_queue_name), "switch_queue_%u", num_src_pad);
    queue = gst_element_factory_make("queue", q_name);
    nvosd = gst_element_factory_make("nvdsosd", osd_name);
    input_selector = gst_element_factory_make("input-selector", input_selector_name);
    videotestsrc = gst_element_factory_make("videotestsrc", videotestsrc_name);
    switch_queue = gst_element_factory_make("queue", switch_queue_name);

    // for rendering the osd output
    if(prop.integrated) {
      sink = gst_element_factory_make("nv3dsink", sink_name);
    } else {
      #ifdef __aarch64__
        sink = gst_element_factory_make ("nv3dsink", sink_name);
      #else
        sink = gst_element_factory_make ("nveglglessink", sink_name);
      #endif
    }

    if (!queue || !nvosd || !sink || !input_selector || !videotestsrc || !switch_queue)
    {
      g_printerr("One element could not be created [demux]. Exiting...\n");
      return -1;
    }

    gst_bin_add_many (GST_BIN(pipeline), queue, nvosd, sink, input_selector, videotestsrc, switch_queue, NULL);
    if (!gst_element_link_many (queue, nvosd, NULL))
    {
       g_printerr ("All elements could not be linked\n");
       return -1;
    }
    /* Link the input selector to the sink */
    if(!gst_element_link(input_selector, sink))
    {
      g_printerr("Input selector and sink could not be linked\n");
      return -1;
    }
    /* Link the test pattern source to its queue */
    if(!gst_element_link(videotestsrc, switch_queue))
    {
      g_printerr("Videotestsrc and switch queue could not be linked\n");
      return -1;
    }
    /* Connect the test pattern path to input selector sink_0 */
    sink_pad = gst_element_request_pad_simple(input_selector,"sink_0");
    src_pad = gst_element_get_static_pad(switch_queue,"src");
    if (gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link videotestsrc path to input selector\n");
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        return -1;
    }
    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);

    /* Connect the primary processing path to input selector sink_1 */
    sink_pad = gst_element_request_pad_simple(input_selector,"sink_1");
    src_pad = gst_element_get_static_pad(nvosd,"src");
    if (gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link nvosd to input selector\n");
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        return -1;
    }

    g_object_set (G_OBJECT(sink), "sync", FALSE, NULL);
    g_object_set (G_OBJECT(sink), "max-lateness", 200000000, NULL);
    g_object_set (G_OBJECT(videotestsrc), "pattern", 2, NULL);
    g_object_set (G_OBJECT(input_selector), "active-pad", sink_pad, NULL);
    g_print ("create_branch: source_no: %u\t pad_name=%s\n", num_src_pad, pad_name);

    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);
    src_pad = gst_element_request_pad_simple (nvstreamdemux, pad_name);
    sink_pad = gst_element_get_static_pad (queue, "sink");

    if (!src_pad)
    {
      g_printerr ("src pad has not been created\n");
      return -1;
    }
    if (!sink_pad)
    {
      g_printerr ("sink pad has not been created\n");
      return -1;
    }
    if (gst_pad_link (src_pad, sink_pad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link. Exiting.\n");
      gst_object_unref(src_pad);
      gst_object_unref(sink_pad);
      return -1;
    }
    /* Add a probe to detect pad deleted events */
    ProbeData *probe_data = g_new0(ProbeData, 1);
    if (!probe_data) {
        g_printerr("Failed to allocate memory for probe data\n");
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        return -1;
    }
    probe_data->input_selector = input_selector;
    probe_data->num_src_pad = num_src_pad;
    gulong probe_id = gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, 
                     demux_src_pad_probe_cb, probe_data, g_free);
    
    if (probe_id == 0) {
        g_printerr("Failed to add probe to source pad\n");
        g_free(probe_data);
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        return -1;
    }

    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);
    num_src_pad++;
    return 0;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

/**
 * Detects new streams and handles them appropriately:
 * 1. Creates a new branch if needed for upcoming streams
 * 2. Processes any previously ended streams
 */
static GstPadProbeReturn
pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);

  /* This event indicates the beginning of a new stream, making it
   * the ideal point to detect and handle new input sources. */
  if (GST_EVENT_TYPE(event) == GST_EVENT_STREAM_START) {
    if(num_src_pad - num_src_pad_used == 1 && num_src_pad < max_batch_size)
    {
      create_branch();
    }
    num_src_pad_used++;

    /* Process any ended streams that could be reused */
    g_mutex_lock(&pipeline_mutex);
    while (!g_queue_is_empty(ended_streams_queue)) {
        guint ended_pad_num = GPOINTER_TO_UINT(g_queue_peek_head(ended_streams_queue));
        gchar input_selector_name[26];
        snprintf(input_selector_name, sizeof(input_selector_name), "input_selector_%u", ended_pad_num);
        GstElement *current_input_selector = gst_bin_get_by_name(GST_BIN(pipeline), input_selector_name);
        if (current_input_selector) {
            GstPad *sink_pad = gst_element_get_static_pad(current_input_selector, "sink_1");
            if (sink_pad) {
                g_object_set(current_input_selector, "active-pad", sink_pad, NULL);
                gst_object_unref(sink_pad);
            }
            gst_object_unref(current_input_selector);
            g_queue_pop_head(ended_streams_queue);
        }
    }
    g_mutex_unlock(&pipeline_mutex);
  }

  return GST_PAD_PROBE_OK;
}

int
main (int argc, char *argv[])
{
  g_mutex_init (&pipeline_mutex);
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  gboolean yaml_config = FALSE;
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Parse inference plugin type */
  yaml_config = (g_str_has_suffix (argv[1], ".yml") ||
          g_str_has_suffix (argv[1], ".yaml"));

  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
  }

  /* Create pipeline and source element */
  pipeline = gst_pipeline_new ("demuxer-pipeline");
  nvmultiurisrcbin = gst_element_factory_make ("nvmultiurisrcbin", "source");
  GstPad *srcpad = NULL;
  if (!pipeline || !nvmultiurisrcbin) {
    g_printerr ("One element could not be created. Exiting.\n");
    goto cleanup;
  }

  /* Create inference engine based on configuration */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
  } else {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  }

  /* Create stream demuxer */
  nvstreamdemux = gst_element_factory_make("nvstreamdemux","nvstreamdemux");

  if (!nvstreamdemux || !pgie ) {
    g_printerr ("One element could not be created. Exiting.\n");
    goto cleanup;
  }

  g_object_set(G_OBJECT(nvmultiurisrcbin), "ip-address", "localhost", NULL);
  if (yaml_config) {
    RETURN_ON_PARSER_ERROR(nvds_parse_multiurisrcbin(nvmultiurisrcbin, argv[1],"source"));
    RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
  }

  /* Add bus message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the main pipeline elements */
  gst_bin_add_many (GST_BIN (pipeline),
      nvmultiurisrcbin, pgie, nvstreamdemux, NULL);

  if (!gst_element_link_many (nvmultiurisrcbin, pgie, nvstreamdemux, NULL)) {
    g_printerr ("Elements could not be linked: Exiting.\n");
    goto cleanup;
  }

  /* Process initial URI list and create branches */
  gchar *uri_list = NULL;
  g_object_get (G_OBJECT(nvmultiurisrcbin), "uri-list", &uri_list, NULL);
  guint num_uri = 0;
  if (uri_list) {
    gchar **uris = g_strsplit(uri_list, ",", -1);
    for (int i = 0; uris[i] != NULL; i++) {
      g_print("URI %d: %s\n", i + 1, uris[i]);
      create_branch();
      num_uri++;
    }
    g_strfreev(uris);
    g_free(uri_list);
  }
  /* If the URI list is initially empty, the application creates a single processing branch
   * in anticipation of the first incoming stream. This ensures that the pipeline is ready to
   * handle dynamically added sources even when no static sources are defined at startup. */
  if (num_uri == 0)
  {
    create_branch();
  }

  /* Add probe for stream start events */
  srcpad = gst_element_get_static_pad(nvmultiurisrcbin, "src");
  if (srcpad) {
    gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        (GstPadProbeCallback)pad_probe_cb, NULL, NULL);
  } else {
    g_printerr("Failed to get source pad from nvmultiurisrcbin\n");
    goto cleanup;
  }
  /* Get maximum batch size */
  g_object_get (G_OBJECT(nvmultiurisrcbin), "max-batch-size", &max_batch_size, NULL);
  ended_streams_queue = g_queue_new();
  /* Set the pipeline to "playing" state */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

cleanup:
  /* Clean up resources */
  g_print ("Cleaning up resources...\n");
  /* Unreference source pad if it was obtained */
  if (srcpad) {
    gst_object_unref(srcpad);
    srcpad = NULL;
  }
  /* Stop and clean up the pipeline */
  if (pipeline) {
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
  }
  /* Free the queue of ended streams */
  if (ended_streams_queue) {
    g_queue_free(ended_streams_queue);
    ended_streams_queue = NULL;
  }
  /* Remove bus watch */
  if (bus_watch_id > 0) {
    g_source_remove(bus_watch_id);
  }
  /* Unreference the main loop */
  if (loop) {
    g_main_loop_unref(loop);
    loop = NULL;
  }
  /* Destroy the mutex */
  g_mutex_clear(&pipeline_mutex);
  return 0;
}
