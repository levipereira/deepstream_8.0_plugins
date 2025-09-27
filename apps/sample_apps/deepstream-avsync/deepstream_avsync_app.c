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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <gst/gstclock.h>
#include <unistd.h>
#include <termios.h>

#include "gstnvdsmeta.h"
//#include "gstnvstreammeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#define SUPPORT_AUDIO
#define MUX_AV

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 33333

//#define TILED_OUTPUT_WIDTH 1280
//#define TILED_OUTPUT_HEIGHT 720
#define TILED_OUTPUT_WIDTH 480
#define TILED_OUTPUT_HEIGHT 360

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "RoadSign"
};

#define FPS_PRINT_INTERVAL 300

static guint cintr = FALSE;

/* Flag which is transmitted alongwith the signal "stream-toggle-on"
 * which is meant to be processed by asr plugin.
 * Use key 'e' to enable and 'd' to disable.
 * By default ASR feature is enabled
 */
static gboolean asr_enable = TRUE;

int rtmp_out = 1;
int rtsp_out = 0;
guint num_sources = 0;
GstElement *pipeline = NULL;
//static struct timeval start_time = { };

//static guint probe_counter = 0;

gchar source_type[5] = { 0 };
typedef struct _Asr_output
{
  guint8 * asr_text;
  guint text_buffer_size;
  GstClockTime text_buf_pts;
} Asr_output;

typedef struct _audio_source_info
{
  int source_index;
  GQueue * queue;
  GMutex mutex;
} audio_source_info;

typedef struct _AppCtx
{
  audio_source_info **source_info;
} Appctx;


/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
void _intr_handler (int signum)
{
  struct sigaction action;

  g_printerr ("User Interrupted.. \n");
  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  cintr = TRUE;
}

/**
 * Loop function to check the status of interrupts.
 * It sends EOS to the pipeline if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (cintr)
  {
      /* Handle the interrupt
       * Send EOS to the pipeline */
      if (!gst_element_send_event (GST_ELEMENT(pipeline),
            gst_event_new_eos()))
        g_print("Interrupted, EOS not sent");

      return FALSE;
  }

  return TRUE;
}
/*
* Function to install custom handler for program interrupt signal.
*/
void _intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

/* tiler_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */


#if 1

GstClockTime
gst_get_current_clock_time (GstElement * element);
GstClockTime
gst_get_current_clock_time (GstElement * element)
{
  GstClock *clock = NULL;
  GstClockTime ret;

  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);

  clock = gst_element_get_clock (element);

  if (!clock) {
    GST_DEBUG_OBJECT (element, "Element has no clock");
    return GST_CLOCK_TIME_NONE;
  }

  ret = gst_clock_get_time (clock);
  gst_object_unref (clock);

  return ret;
}

GstClockTime
gst_get_current_running_time (GstElement * element);
GstClockTime
gst_get_current_running_time (GstElement * element)
{
  GstClockTime base_time, clock_time;

  g_return_val_if_fail (GST_IS_ELEMENT (element), GST_CLOCK_TIME_NONE);

  base_time = gst_element_get_base_time (element);

  if (!GST_CLOCK_TIME_IS_VALID (base_time)) {
    GST_DEBUG_OBJECT (element, "Could not determine base time");
    return GST_CLOCK_TIME_NONE;
  }

  clock_time = gst_get_current_clock_time (element);

  if (!GST_CLOCK_TIME_IS_VALID (clock_time)) {
    return GST_CLOCK_TIME_NONE;
  }

  if (clock_time < base_time) {
    GST_DEBUG_OBJECT (element, "Got negative current running time");
    return GST_CLOCK_TIME_NONE;
  }

  return clock_time - base_time;
}

static GstPadProbeReturn
rtmpsrc_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    GstEvent *event = (GstEvent *) info->data;
    GstClockTime running_time;
    GstSegment *segment;

    if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT)
    {
        running_time = gst_get_current_running_time (pipeline);
        gst_event_parse_segment (event, (const GstSegment **)&segment);
        gst_print ("got segment before segment_base = %"GST_SEGMENT_FORMAT "\n", segment);
        segment->base = running_time;
        gst_print ("got segment after segment_base = %"GST_SEGMENT_FORMAT "\n", segment);
    }

    return GST_PAD_PROBE_OK;
}

static void
decodebin_deep_element_added (GstBin * self, GstBin * sub_bin,
            GstElement * element,  gpointer user_data)
{
   GstElementFactory *factory = gst_element_get_factory (element);
   g_print ("deep element added: %s\n", GST_ELEMENT_NAME(element));


  if (!g_strcmp0(GST_OBJECT_NAME(factory), "rtmpsrc"))
  {
    gst_print ("*****************************SEGMENT EVENT \n");
     GstPad *srcpad = gst_element_get_static_pad (element, "src");
    gst_pad_add_probe(srcpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                        rtmpsrc_sink_pad_buffer_probe, NULL, NULL);
  }
}

#endif

static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    //NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
          //g_print ("Frame Number = %d Number of objects = %d "
            //"Vehicle Count = %d Person Count = %d\n",
            //frame_meta->frame_num, num_rects, vehicle_count, person_count);
#if 0
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
#endif

    }
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
nvosd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsMetaList *display_meta_list = NULL;
    NvDsMetaList *l = NULL;
    NvOSD_TextParams *txt_params = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    display_meta_list = batch_meta->display_meta_pool->full_list;

    int i = 0;
    for (l = display_meta_list; l != NULL; l = l->next)
    {
      display_meta = (NvDsDisplayMeta *)(l->data);
      txt_params = &display_meta->text_params[0];
      txt_params->x_offset = 10;;
      //txt_params->y_offset = 20;//(TILED_OUTPUT_HEIGHT - 100) + (20 * i);
      txt_params->y_offset = 10 + (display_meta->misc_osd_data[0]  * (TILED_OUTPUT_HEIGHT/ num_sources));
      //g_print("In osd sink pad y offset = %d\n", txt_params->y_offset);
      i++;
    }
    return GST_PAD_PROBE_OK;
}

#if 1
static GstPadProbeReturn
tiler_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    Appctx *appctx = (Appctx *)u_data;
    NvDsMetaList * l_frame = NULL;
    NvDsDisplayMeta *display_meta = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    Asr_output *output = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
      audio_source_info *source_info = appctx->source_info[frame_meta->pad_index];

      g_mutex_lock(&source_info->mutex);
      /* check if asr output is present */
      if (g_queue_is_empty(source_info->queue))
      {
        /* no asr output is present, return from here */
        g_mutex_unlock(&source_info->mutex);
        //g_print("\n No output line = %d\n",__LINE__);
        continue;
      }

      output = (Asr_output *)g_queue_peek_head(source_info->queue);
      g_mutex_unlock(&source_info->mutex);

      if (output == NULL)
      {
        //g_print("\n No output line = %d\n",__LINE__);
        continue;
      }

      //g_print("\n\nvideo pts: %" GST_TIME_FORMAT "audio pts: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(frame_meta->buf_pts), GST_TIME_ARGS(output->text_buf_pts));
      //g_print("video pts = %ld audio pts = %ld\n", frame_meta->buf_pts, output->text_buf_pts);
      /* If audio is ahead, do not overlay */
      if (output->text_buf_pts > frame_meta->buf_pts)
      {
        //g_print("func = %s line = %d\n", __func__, __LINE__);
        //g_print("\n No output line = %d\n",__LINE__);
        continue;
      }
      /*
      else if ((GstClockTime)(frame_meta->buf_pts - output->text_buf_pts) > (GstClockTime)3000000000)
      {
        g_mutex_lock(&source_info->mutex);
        output = (Asr_output *)g_queue_pop_head(source_info->queue);
        g_mutex_unlock(&source_info->mutex);
        //g_print("video pts: %" GST_TIME_FORMAT "audio pts: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(frame_meta->buf_pts), GST_TIME_ARGS(output->text_buf_pts));
        //g_print("video pts = %ld audio pts = %ld\n", frame_meta->buf_pts, output->text_buf_pts);

        free(output->asr_text);
        output->asr_text = NULL;
        output->text_buffer_size = 0;
        free(output);
        output = NULL;
        //g_print("func = %s line = %d\n", __func__, __LINE__);
        g_print("\n No output line = %d\n",__LINE__);
        continue;
      }
      */
      else
      {
        NvOSD_TextParams *txt_params = NULL;

        g_mutex_lock(&source_info->mutex);

        guint length = g_queue_get_length (source_info->queue);
        //g_print("queue len = %d\n", length);

        if(length > 1)
        {
          while (length)
          {
            output = (Asr_output *)g_queue_peek_head(source_info->queue);
            Asr_output *next_output = (Asr_output *)g_queue_peek_nth(source_info->queue, 1);
            if(next_output == NULL)
            {
              break;
            }

            if(next_output->text_buf_pts > frame_meta->buf_pts)
            {
              //g_print("Do not remove from the list\n");
              break;
            }
            else
            {
              output = (Asr_output *)g_queue_pop_head(source_info->queue);
              free(output->asr_text);
              output->asr_text = NULL;
              output->text_buffer_size = 0;
              free(output);
              output = NULL;
            }
            length--;
          }
        }

        g_mutex_unlock(&source_info->mutex);

        //g_print("\n\nvideo pts: %" GST_TIME_FORMAT "audio pts: %" GST_TIME_FORMAT "\n", GST_TIME_ARGS(frame_meta->buf_pts), GST_TIME_ARGS(output->text_buf_pts));
        //g_print("video pts = %ld audio pts = %ld\n", frame_meta->buf_pts, output->text_buf_pts);
        if((GstClockTime)(frame_meta->buf_pts - output->text_buf_pts) > (GstClockTime)1000000000)
        {
          g_mutex_lock(&source_info->mutex);
          output = (Asr_output *)g_queue_pop_head(source_info->queue);
          g_mutex_unlock(&source_info->mutex);
          free(output->asr_text);
          output->asr_text = NULL;
          output->text_buffer_size = 0;
          free(output);
          output = NULL;
          continue;
        }

        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;

        txt_params->display_text = g_malloc0(output->text_buffer_size);
        memcpy(txt_params->display_text, output->asr_text, output->text_buffer_size);
        //g_print("\n In tiler sink pad ASR output = %s\n", output->asr_text);
        //g_print("In tiler sink display text = %s\n", txt_params->display_text);
        //g_print("func = %s line = %d\n", __func__, __LINE__);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 100;
        txt_params->y_offset = 100;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 0.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 0;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.6;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        display_meta->misc_osd_data[0] = frame_meta->source_id;

        g_print("\nsource %d: asr text = %s\n", frame_meta->source_id , txt_params->display_text);
        //g_print("Before display frame pts = %ld audio pts = %ld\n", frame_meta->buf_pts, output->text_buf_pts);
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
      }
    }
    return GST_PAD_PROBE_OK;
}
#endif

static GstPadProbeReturn
nvasr_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                           gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  audio_source_info *source_info = (audio_source_info *)u_data;
  int source_index = source_info->source_index;
  //g_print("Source_index = %d\n", source_index);
  source_index = source_index;
#if 1
  guint8 *text_data = NULL;
  int buffer_size = 0;
  GstMapInfo inmap = GST_MAP_INFO_INIT;

  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    g_print ("Unable to map info from buffer\n");
    return GST_FLOW_ERROR;
  }
  text_data = (guint8 *)inmap.data;
  buffer_size = gst_buffer_get_size(buf) + 1;
  //text_data[buffer_size] = '\0';
  buffer_size = buffer_size;
#endif

#if 1
  Asr_output *output = (Asr_output *)g_malloc0(sizeof(Asr_output));
  output->asr_text = (guint8 *)g_malloc0(buffer_size);
  memcpy(output->asr_text, text_data, buffer_size);
  output->text_buf_pts = GST_BUFFER_PTS(buf);
  output->text_buffer_size = buffer_size;
  g_mutex_lock (&source_info->mutex);
  g_queue_push_tail(source_info->queue, (gpointer)output);
  g_mutex_unlock (&source_info->mutex);
  //g_print("In nvasr SRC pad ASR output = %s\n", output->asr_text);
#else
  /* dummy code. Need actual buffer content here */
  gchar source_name[512] = {};
  g_snprintf(source_name, 511, "I_am_source_%u checking here for 1 checking here for 1 checking here for 1 checking here for 1checking here for 1checking here for 1checking here for 1 ", source_index);

  Asr_output *output = (Asr_output *)malloc(sizeof(Asr_output));
  output->asr_text = malloc(512);
  memcpy(output->asr_text, source_name, 512);
  output->text_buf_pts = GST_BUFFER_PTS(buf);
  output->text_buffer_size = 512;
  g_mutex_lock (&source_info->mutex);
  g_queue_push_tail(source_info->queue, (gpointer)output);
  g_mutex_unlock (&source_info->mutex);
#endif
  return GST_PAD_PROBE_OK;
}

#if 0
int input_audio_buffer_count = 0;
static GstPadProbeReturn
nvasr_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                           gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  audio_source_info *source_info = (audio_source_info *)u_data;
  int source_index = source_info->source_index;
  //g_print("Source_index = %d\n", source_index);
  buf = buf;
  source_index = source_index;
  //g_print("Input audio_buffer_count = %d func = %s line = %d\n", input_audio_buffer_count, __func__,__LINE__);
  input_audio_buffer_count++;
  return GST_PAD_PROBE_OK;
}
#endif

//int muxer_cnt = 0;
static GstPadProbeReturn
streammux_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  //g_print("muxer_cnt = %d\n", muxer_cnt++);
  return GST_PAD_PROBE_OK;
}

int input_muxer_cnt = 0;
static GstPadProbeReturn
streammux_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  //g_print("input muxer_cnt = %d\n", input_muxer_cnt++);
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
    case GST_MESSAGE_WARNING:
    {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
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

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_query_caps (decoder_src_pad, NULL);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Get the source bin ghost pad */
    GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "vsrc");
    if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
            decoder_src_pad)) {
      g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
    }
    gst_object_unref (bin_ghost_pad);
  }

#ifdef SUPPORT_AUDIO
 if (!strncmp (name, "audio", 5))
  {
    /* Get the source bin ghost pad */
    GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "asrc");

    if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
          decoder_src_pad))
    {
      g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
    }
    gst_object_unref (bin_ghost_pad);

    gst_element_sync_state_with_parent(source_bin);
  }
#endif

}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "urisourcebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  /* Commented to suppress warnings*/
  /*if (!(g_strcmp0 (source_type, "rtmp"))) {
     g_object_set (G_OBJECT (object), "do-timestamp", 1, NULL);
     g_object_set (G_OBJECT (object), "timeout", 10000, NULL);
  }*/
}

static GstElement *
create_source_bin (guint index, gchar * uri)
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
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);
  g_object_set (G_OBJECT (uri_decode_bin), "async-handling", true, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);
   g_signal_connect (G_OBJECT (uri_decode_bin), "deep-element-added",
      G_CALLBACK (decodebin_deep_element_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("vsrc",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add video ghost pad in source bin\n");
    return NULL;
  }
#ifdef SUPPORT_AUDIO
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("asrc",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add audio ghost pad in source bin\n");
    return NULL;
  }
#endif

  return bin;
}

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
static void
changemode (int dir)
{
  static struct termios oldt, newt;

  if (dir == 1) {
    tcgetattr (STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON);
    tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  } else
    tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
}

/**
  * Loop function to check keyboard inputs and status of the pipeline.
  */
static gboolean
event_thread_func (gpointer arg)
{

  GstElement **nvasr_instances = (GstElement**) arg;

    if (!kbhit())
        return TRUE;

    int c = fgetc (stdin);

    /* Based on keyboard input, "stream-toggle-on" signal
     * alongwith the asr_enable value is emitted.
     */
    switch(c){
      case 'd':
        asr_enable = FALSE;
        g_print ("Got the keyboard interrupt to disable ASR feature.. \n");
        for (int i =0; i< num_sources; i++)
          g_signal_emit_by_name(nvasr_instances[i],"stream-toggle-on",asr_enable);
        break;
      case 'e':
        asr_enable = TRUE;
        g_print ("Got the keyboard interrupt to enable ASR feature.. \n");
        for (int i =0; i< num_sources; i++)
          g_signal_emit_by_name(nvasr_instances[i],"stream-toggle-on",asr_enable);
        break;
      default:
        break;
      }

    return TRUE;
}

static void
print_runtime_commands (void)
{
  g_print ("\nRuntime commands:\n"
      "\td: Disbale ASR feature\n\n" "\te: Enable ASR feaure\n\n");

  g_print
        ("NOTE: By default ASR is enabled. \n\n");
}

int
main (int argc, char *argv[])
{
  Appctx appctx;
  GMainLoop *loop = NULL;
  //GstElement *pipeline = NULL, *streammux = NULL, *pgie = NULL,
  GstElement *streammux = NULL, *pgie = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *queue6, *queue7, *queue8,
      *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL; // *sink = NULL;
  GstElement *rtspoutsinkbin = NULL;

  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

#ifdef SUPPORT_AUDIO
  GstElement *audiomixer = NULL, *nvanr = NULL, *nvasr = NULL;
  GstPad *nvanr_sinkpad = NULL, *audio_srcpad = NULL,
         *amixer_sinkpad = NULL;
  GstPad *nvasr_srcpad = NULL, *nvasr_sinkpad = NULL;
  GstElement *tee = NULL, *sink_asr_pipeline = NULL;
  GstPad *tee_asr_srcpad = NULL, *tee_amixer_srcpad = NULL;
  GstElement *audioconvert = NULL, *capsfilter = NULL;
  const gchar *caps_str = "audio/x-raw, format=(string)S16LE, channels=(int)1";

  GstElement *audioresampler = NULL, *resampler_filter = NULL;
  GstPad *resampler_sinkpad = NULL;
  const gchar *resampler_caps_str = "audio/x-raw, rate=(int)16000";
  GstCaps *filtercaps = NULL;

#endif

#ifdef MUX_AV
  GstElement *audio_convert = NULL, *audio_enc = NULL, *audioparser = NULL, *flvmux = NULL,
         *rtmpsink = NULL;
  GstPad *flvmux_audiopad = NULL;
  GstPad *audioparser_srcpad = NULL;

  GstElement *nvvideoconvert = NULL, *video_enc = NULL, *codecparser = NULL;
  GstPad *flvmux_videopad = NULL, *codecparser_srcpad = NULL;

  GstElement *fakesrc = NULL;
  GstPad *fakesrc_srcpad = NULL;
#endif

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *tiler_src_pad = NULL;
  GstPad *tiler_sink_pad = NULL;
  GstPad *nvosd_sink_pad = NULL;
  //guint i, num_sources;
  guint i;
  guint tiler_rows, tiler_columns;
  //Uncomment below if nvinfer is used as pgie
  //guint pgie_batch_size;
  gchar *rtmp_stream = NULL;
  guint rtsp_port_num = 0;
  guint enc_type = 0;

  /* Check input arguments */
  if (argc < 6)
  {
    g_printerr("Usage: %s <uri1> [uri2] ... [uriN] <Flag to indicate to stream\
    output over rtsp/rtmp server 1:RTSP, 0:RTMP> <RTMP server url :applicable only for RTMP, for RTSP specify port number>\
    enc-type <0:Hardware encoder 1:Software encoder>\n \
    e.g. For RTSP: deepstream-avsync-app file:///uri.mp4 1 8554 enc-type 0\n \
         For RTMP: deepstream-avsync-app file:///uri.mp4 0 rtmp://a.rtmp.youtube.com/live2/abcd-ef1g-1234-5hij-klmn enc-type 0\n",
               argv[0]);
    return -1;
  }
  num_sources = argc - 5;

  /* nvasr_instances contains the asr plugin instances getting used in the app.
   * The "enable-asr" signal is sent on each of these asr instances
   */
  GstElement *nvasr_instances[num_sources];

  snprintf (source_type, 5, "%s", argv[1]);

  rtsp_out = atoi(argv[argc - 4]);

  if (rtsp_out)
  {
    rtmp_out = 0;
    rtsp_port_num = atoi(argv[argc - 3]);
    g_print("\n*** Audio and video output will sent via rtsp server\n");
  }
  else
  {
    rtmp_out = 1;
    rtmp_stream = argv[argc - 3];
    g_print("\n*** Audio video output will sent via rtmp server\n");
    g_print("*** RTMP Server stream will be ready at %s\n\n", rtmp_stream);
  }

  enc_type = atoi(argv[argc - 1]);

  memset(&appctx, 0, sizeof(Appctx));

  appctx.source_info = (audio_source_info **)g_malloc0(sizeof(audio_source_info*) * num_sources);
  if(appctx.source_info == NULL)
  {
    return -1;
  }

  for(i = 0; i < num_sources; i++)
  {
    appctx.source_info[i] = (audio_source_info *)g_malloc0(sizeof(audio_source_info ));
    appctx.source_info[i]->queue = g_queue_new ();
    g_mutex_init (&appctx.source_info[i]->mutex);
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  _intr_setup ();

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-avsync-pipeline");

  g_timeout_add (400, check_for_interrupt, NULL);
  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  /* Create audio mixer instance to mix audio from multiple sources. */
  audiomixer = gst_element_factory_make("audiomixer", "audio-mixer");

  /* Create audio mixer instance to mix audio from multiple sources. */
  fakesrc = gst_element_factory_make("fakesrc", "fakesrc");

  if (!pipeline || !streammux || !audiomixer || !fakesrc) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (!(g_strcmp0 (source_type, "rtmp")))
  {
    g_object_set(G_OBJECT(audiomixer), "latency",    250000000, NULL);
    g_object_set(G_OBJECT(streammux), "max-latency", 250000000, NULL);
  }

  /*if (!(g_strcmp0 (source_type, "rtsp")))
  {
    g_object_set(G_OBJECT(streammux), "max-latency", 2000000000, NULL);
  }*/

  gst_bin_add_many (GST_BIN (pipeline), streammux, audiomixer, fakesrc, NULL);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    gchar anr_name[16] = { };
    gchar asr_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "vsrc");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    if (i == 0)
    {
      gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                        streammux_sink_pad_buffer_probe, NULL, NULL);
    }
    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

#ifdef SUPPORT_AUDIO
   /* Pipeline: decoder->nvanr->audiomixer->sink */
    g_snprintf (anr_name, 15, "anr_%u", i);
    //nvanr = gst_element_factory_make ("identity", anr_name);

    /* The layout is non-interleaved in open source avdec_aac
      for Ubuntu 20.04. Need to add audioconvert to make the layout
      as interleaved */
    nvanr = gst_element_factory_make ("audioconvert", anr_name);

    if (!nvanr)
    {
      g_printerr("nvanr could not be created. Exiting.\n");
      return -1;
    }

    g_snprintf (asr_name, 15, "asr_%u", i);

    tee = gst_element_factory_make ("tee", NULL);

    if (!tee)
    {
      g_printerr("tee could not be created. Exiting.\n");
      return -1;
    }

    audioconvert = gst_element_factory_make ("audioconvert", NULL);

    if (!audioconvert)
    {
      g_printerr("audioconvert could not be created. Exiting.\n");
      return -1;
    }

    capsfilter = gst_element_factory_make ("capsfilter", NULL);

    if (!capsfilter)
    {
      g_printerr("capsfilter could not be created. Exiting.\n");
      return -1;
    }

    filtercaps = gst_caps_from_string(caps_str);
    g_object_set (capsfilter, "caps", filtercaps, NULL);
    gst_caps_unref (filtercaps);

    audioresampler = gst_element_factory_make ("audioresample", NULL);

    if (!audioresampler)
    {
      g_printerr("audioresampler could not be created. Exiting.\n");
      return -1;
    }

    resampler_filter = gst_element_factory_make ("capsfilter", NULL);

    if (!resampler_filter)
    {
      g_printerr("resamplerfilter could not be created. Exiting.\n");
      return -1;
    }

    filtercaps = gst_caps_from_string(resampler_caps_str);
    g_object_set (resampler_filter, "caps", filtercaps, NULL);
    gst_caps_unref (filtercaps);

    nvasr = gst_element_factory_make ("nvdsasr", asr_name);

    g_object_set (G_OBJECT (nvasr), "config-file", "riva_asr_grpc_jasper_conf.yml", NULL);
    g_object_set (G_OBJECT (nvasr), "customlib-name", "libnvds_riva_asr_grpc.so", NULL);
    g_object_set (G_OBJECT (nvasr), "create-speech-ctx-func", "create_riva_asr_grpc_ctx", NULL);

    nvasr_instances[i] = nvasr;

    if (!nvasr)
    {
      g_printerr("nvanr could not be created. Exiting.\n");
      return -1;
    }

    sink_asr_pipeline = gst_element_factory_make ("fakesink", NULL);

    if (!sink_asr_pipeline)
    {
      g_printerr("sink element for asr pipeline could not be created. Exiting.\n");
      return -1;
    }

    g_object_set(G_OBJECT(sink_asr_pipeline), "async", false, NULL);
    g_object_set(G_OBJECT(sink_asr_pipeline), "sync", false, NULL);

    gst_bin_add_many (GST_BIN (pipeline), nvanr, tee, audioresampler, resampler_filter,
     audioconvert, capsfilter, nvasr, sink_asr_pipeline, NULL);

    if( !gst_element_link_many(nvanr, tee, NULL))
    {
      g_print("Failed to link nvanr and nvasr elements\n");
      return -1;
    }

    audio_srcpad = gst_element_get_static_pad (source_bin, "asrc");
    if (!audio_srcpad) {
      g_printerr ("Failed to get audio src pad of source bin. Exiting.\n");
      return -1;
    }

    nvanr_sinkpad = gst_element_get_static_pad(nvanr, "sink");
    if (!nvanr_sinkpad)
    {
      g_printerr("nvanr sink pad failed. \n");
      return -1;
    }

    if (gst_pad_link(audio_srcpad, nvanr_sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link audio source bin to nvanr. Exiting.\n");
      return -1;
    }
    gst_object_unref(nvanr_sinkpad);
    gst_object_unref(audio_srcpad);

    tee_asr_srcpad = gst_element_request_pad_simple(tee, "src_%u");
    tee_amixer_srcpad = gst_element_request_pad_simple(tee, "src_%u");

    if (!tee_asr_srcpad || ! tee_amixer_srcpad)
    {
      g_printerr("tee element's src pad request failed. Exiting.\n");
      return -1;
    }

    amixer_sinkpad = gst_element_request_pad_simple(audiomixer, pad_name);
    if (!amixer_sinkpad)
    {
      g_printerr("audio-mixer request sink pad failed. Exiting.\n");
      return -1;
    }

    nvasr_sinkpad = gst_element_get_static_pad(nvasr, "sink");
    if (!nvasr_sinkpad)
    {
      g_printerr("nvasr sink pad failed. \n");
      return -1;
    }

    nvasr_srcpad = gst_element_get_static_pad(nvasr, "src");
    if (!nvasr_srcpad)
    {
      g_printerr("nvasr src pad failed. \n");
      return -1;
    }

    if (gst_pad_link(tee_amixer_srcpad, amixer_sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link audiomixer sink pad and tee srcpad. Exiting.\n");
      return -1;
    }

    resampler_sinkpad = gst_element_get_static_pad(audioresampler, "sink");
    if (!resampler_sinkpad)
    {
      g_printerr("resampler sink pad failed. \n");
      return -1;
    }

    if (gst_pad_link(tee_asr_srcpad, resampler_sinkpad) != GST_PAD_LINK_OK)
    {
      g_printerr("Failed to link tee srcpad and resampler sinkpad. Exiting.\n");
      return -1;
    }

    if( !gst_element_link_many(audioresampler, resampler_filter, audioconvert, capsfilter, nvasr, sink_asr_pipeline, NULL))
    {
      g_print("Failed to link nvanr and nvasr elements\n");
      return -1;
    }

    appctx.source_info[i]->source_index = i;
    gst_pad_add_probe (nvasr_srcpad, GST_PAD_PROBE_TYPE_BUFFER,
        nvasr_src_pad_buffer_probe, (gpointer)appctx.source_info[i], NULL);

#if 0
    gst_pad_add_probe (nvasr_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
        nvasr_sink_pad_buffer_probe, (gpointer)appctx.source_info[i], NULL);
#endif

    gst_object_unref(nvasr_srcpad);
    gst_object_unref(amixer_sinkpad);
#endif
  }

#if 1
  gchar pad_name[16] = {};
  g_snprintf(pad_name, 15, "sink_%u", i + 1);

  if (!(g_strcmp0 (source_type, "file")) || !(g_strcmp0 (source_type, "rtmp")))
  {
    g_object_set(G_OBJECT(fakesrc), "is_live", true, NULL);
  }
  else if (!(g_strcmp0 (source_type, "rtsp")))
  {
    g_object_set(G_OBJECT(fakesrc), "is_live", false, NULL);
  }

  g_object_set(G_OBJECT(fakesrc), "num-buffers", 0, NULL);

  fakesrc_srcpad = gst_element_get_static_pad(fakesrc, "src");
  if (!fakesrc_srcpad)
  {
    g_printerr("fakesrc src pad failed. \n");
    return -1;
  }

  amixer_sinkpad = gst_element_request_pad_simple(audiomixer, pad_name);
  if (!amixer_sinkpad)
  {
    g_printerr("audio-mixer request sink pad failed. Exiting.\n");
    return -1;
  }
  if (gst_pad_link(fakesrc_srcpad, amixer_sinkpad) != GST_PAD_LINK_OK)
  {
    g_printerr("Failed to link audiomixer sink pad and anr srcpad. Exiting.\n");
    return -1;
  }
#endif

#ifdef MUX_AV
if(rtmp_out)
{
  GstElement *aq1, *aq2, *aq3, *aq4, *aq5;
  aq1 = gst_element_factory_make ("queue", "aqueue1");
  aq2 = gst_element_factory_make ("queue", "aqueue2");
  aq3 = gst_element_factory_make ("queue", "aqueue3");
  aq4 = gst_element_factory_make ("queue", "aqueue4");
  aq5 = gst_element_factory_make ("queue", "aqueue5");
  /*create audio encode and mux  pipeline*/
 /* create audio converter component to convert decoder audio data */
  audio_convert = gst_element_factory_make ("audioconvert", "audioconverter");

  /* create audio encoder to encode audio data */
  audio_enc = gst_element_factory_make ("avenc_aac", "audio encoder");
  /* create nvvideo converter*/
  audioparser = gst_element_factory_make ("aacparse", "aacparser");

  /* create audio encoder to encode audio data */
  flvmux = gst_element_factory_make ("flvmux", "flvmux");

  rtmpsink = gst_element_factory_make("rtmpsink", "rtmp-sink");
  //rtmpsink = gst_element_factory_make("fakesink", "rtmp-sink");

  if (!audio_convert|| !audio_enc || !flvmux || !audioparser || !rtmpsink) {
    g_printerr ("could not create audio converter or audio encoder. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (flvmux), "streamable", true, NULL);
  g_object_set (G_OBJECT (rtmpsink), "location", rtmp_stream, NULL);
  g_object_set (G_OBJECT (rtmpsink), "async", false, NULL);
  g_object_set (G_OBJECT (rtmpsink), "sync", true, NULL);

  gst_bin_add_many (GST_BIN (pipeline), audio_convert, audio_enc, audioparser, flvmux, rtmpsink, aq1, aq2, aq3, aq4, aq5, NULL);

  if (!gst_element_link_many(audiomixer, aq1, audio_convert, aq2, audio_enc, aq3, audioparser, NULL))
  {
    g_printerr("Audio Elements could not be linked. Exiting.\n");
    return -1;
  }

  audioparser_srcpad = gst_element_get_static_pad(audioparser, "src");
  if (!audioparser_srcpad) {
    g_printerr ("src pad of audio encoder failed. \n");
    return -1;
  }

  flvmux_audiopad = gst_element_request_pad_simple(flvmux, "audio");
  if (!flvmux_audiopad) {
    g_printerr ("sink pad of flv muxer failed. \n");
    return -1;
  }

  if (gst_pad_link (audioparser_srcpad, flvmux_audiopad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link audio source bin to tee\n");
    return -1;
  }

  if (!gst_element_link_many(flvmux, aq4, rtmpsink, NULL))
  {
    g_printerr("flvmuxer and rtmpsink could not be linked. Exiting.\n");
    return -1;
  }

  gst_object_unref(audioparser_srcpad);
  gst_object_unref(flvmux_audiopad);
}
#endif

if(rtsp_out)
{
  rtspoutsinkbin = gst_element_factory_make ("nvrtspoutsinkbin", "nvvideo-renderer");
  /* Commented below line to suppress warning*/
  // g_object_set (G_OBJECT (rtspoutsinkbin), "async", false, NULL);
  g_object_set (G_OBJECT (rtspoutsinkbin), "sync", true, NULL);
  g_object_set (G_OBJECT (rtspoutsinkbin), "bitrate", 768000, NULL);
  g_object_set (G_OBJECT (rtspoutsinkbin), "rtsp-port", rtsp_port_num, NULL);
  g_object_set (G_OBJECT (rtspoutsinkbin), "enc-type", enc_type, NULL);
  g_object_set (G_OBJECT (rtspoutsinkbin), "audio-codec-type", 2, NULL);//use ac3 audio codec

  gst_bin_add_many (GST_BIN (pipeline), rtspoutsinkbin, NULL);
  if (!gst_element_link_many(audiomixer, rtspoutsinkbin, NULL))
  {
    g_printerr("Audio Elements could not be linked. Exiting.\n");
    return -1;
  }
}

  /* Use nvinfer to infer on batched frame. */
  //pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  pgie = gst_element_factory_make ("identity", "primary-nvinference-engine");

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");
  queue6 = gst_element_factory_make ("queue", "queue6");
  queue7 = gst_element_factory_make ("queue", "queue7");
  queue8 = gst_element_factory_make ("queue", "queue8");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  //sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
#endif
  //sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  //sink = gst_element_factory_make ("nvrtspoutsinkbin", "nvvideo-renderer");

  if (!pgie || !tiler || !nvvidconv || !nvosd) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "sync-inputs", true, NULL);

  if (!use_new_mux) {
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }
  /* Configure the nvinfer element using the nvinfer config file. */
  /* Commented to suppress warnings, no need to set config-file-path
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "dstest3_pgie_config.txt", NULL);*/

  /* Override the batch-size set in the config file with the number of sources. */
  /* Commented to suppress warnings, no need to set batch-size
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }*/

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, tiler, queue3,
      nvvidconv, queue4, nvosd, queue5, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, tiler, queue3,
        nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, tiler, queue3,
    nvvidconv, queue4, nvosd, queue5, queue6, queue7, queue8, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  //if (!gst_element_link_many (streammux, queue1, pgie, queue2, tiler, queue3,
    //    nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, tiler, queue3,
        nvvidconv, queue4, nvosd, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif


#ifdef MUX_AV
if(rtmp_out)
{
  GstElement *swenc_caps = NULL;
  /* create nvvideo converter*/
  nvvideoconvert = gst_element_factory_make ("nvvideoconvert", "nvvideonverter1");

  /* create nvvideo converter*/
  if (!enc_type) {
    /* use hardware encoder */
    video_enc = gst_element_factory_make ("nvv4l2h264enc", "video_encoder");
  }
  else {
    swenc_caps =  gst_element_factory_make ("capsfilter", NULL);

    GstCaps *enc_caps = NULL;

    enc_caps = gst_caps_from_string ("video/x-h264, profile=(string)baseline");

    g_object_set (G_OBJECT (swenc_caps), "caps", enc_caps, NULL);
    gst_caps_unref (enc_caps);

    gst_bin_add (GST_BIN (pipeline), swenc_caps);

    if (!swenc_caps) {
      g_printerr ("One element in video soft encode pipeline could not be created. Exiting.\n");
      return -1;
    }

    /* use software encoder */
    video_enc = gst_element_factory_make ("x264enc", "video_encoder");

    g_object_set (G_OBJECT (video_enc), "tune", 4, NULL);

  }

  /* create nvvideo converter*/
  codecparser = gst_element_factory_make ("h264parse", "h264parser");

  if (!nvvideoconvert || !video_enc || !codecparser) {
    g_printerr ("One element in video encode pipeline could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add_many(GST_BIN(pipeline), nvvideoconvert, video_enc, codecparser, NULL);

  if(enc_type) {
    if (!gst_element_link_many (nvosd, queue6, nvvideoconvert, queue7, video_enc, swenc_caps, queue8, codecparser, NULL)) {
      g_printerr ("Video soft encode pipeline Elements could not be linked. Exiting.\n");
      return -1;
    }
  }
  else {
    if (!gst_element_link_many (nvosd, queue6, nvvideoconvert, queue7, video_enc, queue8, codecparser, NULL)) {
      g_printerr ("Video encode pipeline Elements could not be linked. Exiting.\n");
      return -1;
    }
  }

  codecparser_srcpad = gst_element_get_static_pad(codecparser, "src");
  if (!codecparser_srcpad) {
    g_printerr ("src pad of codec parser failed. \n");
    return -1;
  }

  flvmux_videopad = gst_element_request_pad_simple(flvmux, "video");
  if (!flvmux_videopad) {
    g_printerr ("video sink pad of flv muxer failed. \n");
    return -1;
  }

  if (gst_pad_link (codecparser_srcpad, flvmux_videopad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link codecparser to flvmux\n");
    return -1;
  }

  gst_object_unref(codecparser_srcpad);
  gst_object_unref(flvmux_videopad);
}
#endif

if(rtsp_out)
{
  if (!gst_element_link_many(nvosd, rtspoutsinkbin, NULL))
  {
    g_printerr("Audio Elements could not be linked. Exiting.\n");
    return -1;
  }
}

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  tiler_src_pad = gst_element_get_static_pad (pgie, "src");
  if (!tiler_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (tiler_src_pad);


if(1)
{
  tiler_sink_pad = gst_element_get_static_pad (tiler, "sink");
  if (!tiler_sink_pad)
    g_print ("Unable to get tiler sink pad\n");
  else
    gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_sink_pad_buffer_probe, (gpointer)&appctx, NULL);
  gst_object_unref (tiler_sink_pad);
}

if(1)
{
  nvosd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!nvosd_sink_pad)
    g_print ("Unable to get nvosd sink pad\n");
  else
    gst_pad_add_probe (nvosd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        nvosd_sink_pad_buffer_probe, (gpointer)&appctx, NULL);
  gst_object_unref (nvosd_sink_pad);
}

  GstPad * streammux_src_pad = gst_element_get_static_pad (streammux, "src");
  if (!streammux_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (streammux_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        streammux_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (streammux_src_pad);

  print_runtime_commands();
  changemode(1);
  g_timeout_add (40, event_thread_func, nvasr_instances);// to check for keyboard inputs

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 1]);
  }
  g_print ("\n");
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

 #if 1
  for(i = 0; i < num_sources; i++)
  {
    Asr_output *output = NULL;
    g_mutex_clear (&appctx.source_info[i]->mutex);

    while(g_queue_get_length(appctx.source_info[i]->queue))
    {
      output = (Asr_output *)g_queue_pop_head(appctx.source_info[i]->queue);
      free(output->asr_text);
      output->asr_text = NULL;
      output->text_buffer_size = 0;
      free(output);
      output = NULL;
    }

    g_queue_free(appctx.source_info[i]->queue);
    free(appctx.source_info[i]);
    appctx.source_info[i] = NULL;
  }

  free(appctx.source_info);
  appctx.source_info = NULL;

 #endif
  g_main_loop_unref (loop);
  return 0;
}
