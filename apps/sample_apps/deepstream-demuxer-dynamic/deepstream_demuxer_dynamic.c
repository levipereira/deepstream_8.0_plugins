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
#include <math.h>
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

#define MEMORY_FEATURES "memory:NVMM"

GMainLoop *loop = NULL;
GstElement *pipeline = NULL, *bin1 = NULL, *nvmultiurisrcbin = NULL, *pgie = NULL, *tee = NULL, *queue_tee = NULL,
           *queue = NULL, *tiler = NULL, *nvosd = NULL, *sink = NULL, *nvstreamdemux = NULL, *bin2 = NULL,
           *vidconvert = NULL, *displaysink = NULL;
guint num_src_pad = 0;
guint num_uri = 0;
guint max_batch_size = 0;
gboolean flag = FALSE;

GMutex pipeline_mutex;
struct cudaDeviceProp prop;

/* Structure to hold input selector and source pad information */
typedef struct {
    GstElement *input_selector;
    guint num_src_pad;
} ProbeData;

static GstPadProbeReturn
demux_src_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);

/* Queue to track streams that have ended */
GQueue *ended_streams_queue = NULL;

/**
 * Creates a new processing branch for a stream from nvstreamdemux.
 * This function sets up a pipeline branch consisting of:
 * nvstreamdemux_src_pad -> queue -> nvdsosd -> input_selector (sink_1)
 *                                               /
 * videotestsrc -> switch_queue --------------> / (sink_0)
 *                                             /
 *                           input_selector ---> displaysink
 * It is called for each new source stream to create a dedicated
 * processing path for on-screen display and output.
 */
static int
create_branch(void)
{
    GstElement *queue_br = NULL, *nvdsosd = NULL, *input_selector = NULL, *videotestsrc = NULL, *switch_queue = NULL;
    gchar q_name[16], osd_name[16], sink_name[16], pad_name[16], input_selector_name[26], videotestsrc_name[16], switch_queue_name[16];
    snprintf (q_name, sizeof(q_name), "queue_%u", num_src_pad);
    snprintf (osd_name, sizeof(osd_name), "nvosd_%u", num_src_pad);
    snprintf (sink_name, sizeof(sink_name), "sink_%u", num_src_pad);
    snprintf (pad_name, sizeof(pad_name), "src_%u", num_src_pad);
    snprintf (input_selector_name, sizeof(input_selector_name), "input_selector_%u", num_src_pad);
    snprintf (videotestsrc_name, sizeof(videotestsrc_name), "videotestsrc_%u", num_src_pad);
    snprintf (switch_queue_name, sizeof(switch_queue_name), "switch_queue_%u", num_src_pad);

    queue_br = gst_element_factory_make("queue", q_name);
    nvdsosd = gst_element_factory_make("nvdsosd", osd_name);
    input_selector = gst_element_factory_make("input-selector", input_selector_name);
    videotestsrc = gst_element_factory_make("videotestsrc", videotestsrc_name);
    switch_queue = gst_element_factory_make("queue", switch_queue_name);
    // for rendering the osd output
    if(prop.integrated) {
      displaysink = gst_element_factory_make("nv3dsink", sink_name);
    } else {
      #ifdef __aarch64__
        displaysink = gst_element_factory_make ("nv3dsink", sink_name);
      #else
        displaysink = gst_element_factory_make ("nveglglessink", sink_name);
      #endif
    }
    if (!queue || !nvdsosd || !displaysink || !input_selector || !videotestsrc || !switch_queue)
    {
      g_printerr("One element could not be created [demux]. Exiting...\n");
      return -1;
    }
    gst_bin_add_many(GST_BIN(bin2), queue_br, nvdsosd, displaysink, input_selector, videotestsrc,switch_queue, NULL);
    if (!gst_element_link_many(queue_br, nvdsosd, NULL))
    {
       g_printerr("All elements could not be linked...[3]\n");
       return -1;
    }
    g_object_set (displaysink, "sync", FALSE, NULL);
    g_object_set(videotestsrc, "pattern", 2, NULL);
    g_print ("create_branch: source_no: %u\t pad_name=%s\n",num_src_pad, pad_name);


    if(!gst_element_link(input_selector, displaysink))
    {
        g_printerr("Failed to link input_selector and displaysink...\n");
        return -1;
    }
    if(!gst_element_link(videotestsrc, switch_queue))
    {
      g_printerr("Failed to link videotestsrc and switch_queue...\n");
      return -1;
    }

    GstPad *src_pad = NULL, *sink_pad = NULL;
    src_pad = gst_element_request_pad_simple (nvstreamdemux, pad_name);
    sink_pad = gst_element_get_static_pad (queue_br, "sink");
    if(!src_pad || !sink_pad)
    {
        g_printerr("one of the pads is not created\n");
        return -1;
    }
    if (gst_pad_link (src_pad,sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link demuxer and queue...\n");
        return -1;
    }

    g_object_unref(src_pad);
    g_object_unref(sink_pad);

    src_pad = gst_element_get_static_pad(switch_queue, "src");
    sink_pad = gst_element_request_pad_simple(input_selector, "sink_0");
    if(!src_pad || !sink_pad)
    {
        g_printerr("one of the pads is not created for videotestsrc and input_selector\n");
        return -1;
    }
    if (gst_pad_link (src_pad,sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link videotestsrc and input_selector...\n");
        return -1;
    }

    g_object_unref(src_pad);
    g_object_unref(sink_pad);

    src_pad = gst_element_get_static_pad(nvdsosd, "src");
    sink_pad = gst_element_request_pad_simple(input_selector, "sink_1");
    if(!src_pad || !sink_pad)
    {
        g_printerr("one of the pads is not created for nvdsosd and input_selector\n");
        return -1;
    }
    if (gst_pad_link (src_pad,sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Failed to link nvdsosd and input_selector...\n");
        return -1;
    }

    g_object_set(input_selector, "active-pad", sink_pad, NULL);
    // Create and initialize the ProbeData structure
    ProbeData *probe_data = g_new(ProbeData, 1);
    probe_data->input_selector = input_selector;
    probe_data->num_src_pad = num_src_pad;

    g_object_unref(src_pad);
    g_object_unref(sink_pad);

    // Add a probe to demuxer srcpad to handle PAD_DELETED event once the stream is removed
    src_pad = gst_element_get_static_pad(nvstreamdemux, pad_name);
    gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, demux_src_pad_probe_cb, probe_data, g_free);

    g_object_unref(src_pad);
    num_src_pad++;
    return 0;
}

/**
 * Creates a new bin (bin2) containing a queue and nvstreamdemux.
 * This bin is then linked to the tee element in bin1.
 */
static int
add_new_bin(void) {
    g_mutex_lock (&pipeline_mutex);
    bin2 = gst_bin_new ("new_branch_bin");
    queue_tee = gst_element_factory_make("queue", "queue_tee");
    nvstreamdemux = gst_element_factory_make("nvstreamdemux", "demuxer");
    if (!queue_tee || !nvstreamdemux) {
        g_printerr ("One of the elements is not created");
    }
    gst_bin_add_many(GST_BIN(bin2), queue_tee, nvstreamdemux, NULL);
    gst_element_link_many (queue_tee, nvstreamdemux, NULL);
    /** Add bin2 to the main pipeline */
    gst_bin_add_many (GST_BIN(pipeline), bin2, NULL);

    GstPad *tee_src_pad = gst_element_request_pad_simple(tee, "src_1");
    GstPad *queue_sink_pad = gst_element_get_static_pad(queue_tee, "sink");

    GstPad* ghost_src_pad = gst_ghost_pad_new("src", tee_src_pad);
    gst_pad_set_active(ghost_src_pad, TRUE);
    if(!ghost_src_pad)
    {
        g_printerr("src_pad is not created..\n");
        return -1;
    }
    gst_element_add_pad (bin1, ghost_src_pad);
    GstPad *ghost_sink_pad = gst_ghost_pad_new("sink", queue_sink_pad);
    gst_pad_set_active (ghost_sink_pad, TRUE);
    gst_element_add_pad (bin2, ghost_sink_pad);
    if(gst_pad_link (ghost_src_pad, ghost_sink_pad) != GST_PAD_LINK_OK)
    {
        g_printerr("Tee and queue are not linked...\n");
        return -1;
    }
    gst_element_sync_state_with_parent (bin2);

    g_mutex_unlock (&pipeline_mutex);
  return 0;
}

static GstPadProbeReturn
src_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    /* This event indicates the beginning of a new stream, making it
    * the ideal point to detect and handle new input sources. */
    if(GST_EVENT_TYPE(event) == GST_EVENT_STREAM_START)
    {
        if (num_uri==1 && flag==FALSE)
        {
            flag = TRUE;
            add_new_bin();
        }
        if (flag && num_src_pad < max_batch_size)
            create_branch();

        g_mutex_lock(&pipeline_mutex);
        if (num_uri > 0)
            num_uri--;
        while (!g_queue_is_empty(ended_streams_queue)) {
            guint ended_pad_num = GPOINTER_TO_UINT(g_queue_peek_head(ended_streams_queue));
            gchar input_selector_name[26];
            snprintf(input_selector_name, sizeof(input_selector_name), "input_selector_%u", ended_pad_num);
            GstElement *current_input_selector = gst_bin_get_by_name(GST_BIN(bin2), input_selector_name);
            if (current_input_selector) {
                GstPad *sink_pad = gst_element_get_static_pad(current_input_selector, "sink_1");
                if (sink_pad) {
                    g_object_set(current_input_selector, "active-pad", sink_pad, NULL);
                    gst_object_unref(sink_pad);
                }
                gst_object_unref(current_input_selector);
                // Remove the processed pad number from the queue
                g_queue_pop_head(ended_streams_queue);
            }
        }
        g_mutex_unlock(&pipeline_mutex);
    }
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
demux_src_pad_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    ProbeData *probe_data = (ProbeData *)user_data;
    GstElement * input_selector = probe_data->input_selector;
    guint num_src_pad = probe_data->num_src_pad;


    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
    {
        // Process all ended streams in the queue
        g_mutex_lock(&pipeline_mutex);
        while (!g_queue_is_empty(ended_streams_queue)) {
            guint ended_pad_num = GPOINTER_TO_UINT(g_queue_peek_head(ended_streams_queue));
            gchar input_selector_name[26];
            snprintf(input_selector_name, sizeof(input_selector_name), "input_selector_%u", ended_pad_num);
            GstElement *current_input_selector = gst_bin_get_by_name(GST_BIN(bin2), input_selector_name);
            if (current_input_selector) {
                GstPad *sink_pad = gst_element_get_static_pad(current_input_selector, "sink_1");
                if (sink_pad) {
                    g_object_set(current_input_selector, "active-pad", sink_pad, NULL);
                    gst_object_unref(sink_pad);
                }
                gst_object_unref(current_input_selector);
                // Remove the processed pad number from the queue
                g_queue_pop_head(ended_streams_queue);
            }
        }

        // Set the current input selector to show videotestsrc
        GstPad *sink_pad = gst_element_get_static_pad(input_selector, "sink_0");
        g_object_set(input_selector, "active-pad", sink_pad, NULL);
        g_queue_push_tail(ended_streams_queue, GUINT_TO_POINTER(num_src_pad));

        g_mutex_unlock(&pipeline_mutex);
    }
    return GST_PAD_PROBE_OK;
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

int main(int argc, char *argv[])
{
    g_mutex_init (&pipeline_mutex);
    GstBus *bus = NULL;
    guint bus_watch_id;
    gboolean yaml_config = FALSE;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
    int current_device = -1;
    cudaGetDeviceProperties(&prop,current_device);

    /* Check input arguments */
    if(argc != 2)
    {
        g_printerr ("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init(&argc,&argv);
    loop = g_main_loop_new (NULL,FALSE);

    /* Parse inference plugin type */
    yaml_config = (g_str_has_suffix(argv[1],".yml") || g_str_has_suffix(argv[1],".yaml"));
    if(yaml_config)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type,argv[1],"primary-gie"));
    }


    pipeline = gst_pipeline_new ("demuxer-pipeline");
    bin1 = gst_bin_new ("bin1");
    nvmultiurisrcbin = gst_element_factory_make ("nvmultiurisrcbin", "source");

    if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
        pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
    } else {
        pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    }
    tee = gst_element_factory_make ("tee", "tee");

    if(!pipeline || !nvmultiurisrcbin || !pgie || !tee)
    {
        g_printerr("One element could not be created..\n");
        return -1;
    }

    if(yaml_config)
    {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie,argv[1],"primary-gie"));
        RETURN_ON_PARSER_ERROR(nvds_parse_multiurisrcbin(nvmultiurisrcbin,argv[1],"source"));
    }

    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    gst_bin_add_many (GST_BIN(bin1), nvmultiurisrcbin, pgie, tee, NULL);
    if (!gst_element_link_many (nvmultiurisrcbin, pgie, tee, NULL))
    {
        g_printerr ("Elements could not be linked.\n");
        return -1;
    }

    queue = gst_element_factory_make ("queue", "queue");
    tiler = gst_element_factory_make ("nvmultistreamtiler", "tiler");
    /* There is a known issue where bbox are incorrect in case were demuxer & tiler are used along with tee.
      WAR: to use videoconvert as identity element for correct bbox */
    vidconvert = gst_element_factory_make ("identity", "vidconv_identity");
    nvosd = gst_element_factory_make ("nvdsosd", "nvosd");
    if(prop.integrated) {
      sink = gst_element_factory_make("nv3dsink", "sink");
    } else {
      #ifdef __aarch64__
        sink = gst_element_factory_make ("nv3dsink", "sink");
      #else
        sink = gst_element_factory_make ("nveglglessink", "sink");
      #endif
    }

    if (!queue || !tiler || !nvosd || !sink || !vidconvert)
    {
        g_printerr("One element could not be created...[2]\n");
        return -1;
    }
    gst_bin_add_many (GST_BIN(bin1), queue, vidconvert, tiler, nvosd, sink, NULL);
    if (!gst_element_link_many (queue, vidconvert, tiler, nvosd, sink, NULL))
    {
        g_printerr ("Element could not be linked...[2]\n");
        return -1;
    }

    GstPad *tee_srcpad = gst_element_request_pad_simple (tee, "src_0");
    GstPad *queue_sinkpad = gst_element_get_static_pad (queue,"sink");

    if (gst_pad_link (tee_srcpad, queue_sinkpad) != GST_PAD_LINK_OK)
    {
        g_printerr("Pads could not be linked...\n");
        return -1;
    }
    gst_object_unref (tee_srcpad);
    gst_object_unref (queue_sinkpad);

    gchar *uri_list = NULL;
    g_object_get (G_OBJECT(nvmultiurisrcbin), "uri-list", &uri_list, NULL);
    if (uri_list) {
        gchar **uris = g_strsplit(uri_list, ",", -1);
        for (int i = 0; uris[i] != NULL; i++) {
            g_print ("URI %d: %s\n", i + 1, uris[i]);
            num_uri++;
        }
    }

    GstPad *src_pad = gst_element_get_static_pad (nvmultiurisrcbin, "src");
    gst_pad_add_probe (src_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                      (GstPadProbeCallback) src_pad_probe_cb, NULL, NULL);
    gst_bin_add (GST_BIN(pipeline), bin1);
    /* If the URI list is initially empty, the application creates a single processing branch
    * in anticipation of the first incoming stream. This ensures that the pipeline is ready to
    * handle dynamically added sources even when no static sources are defined at startup. */
    if (num_uri == 0)
    {
        add_new_bin();
        create_branch();
        flag = TRUE;
    }
    else
    {
        num_src_pad = num_uri;
    }
    RETURN_ON_PARSER_ERROR(nvds_parse_tiler(tiler, argv[1], "tiler"));
    ended_streams_queue = g_queue_new();
    g_object_get (G_OBJECT(nvmultiurisrcbin), "max-batch-size", &max_batch_size, NULL);
    g_print ("Using file: %s\n", argv[1]);
    /* Set the pipeline to "playing" state */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);
    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);

    // Clean up the queue
    if (ended_streams_queue) {
        g_queue_free(ended_streams_queue);
        ended_streams_queue = NULL;
    }

    return 0;
}