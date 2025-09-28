/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <string.h>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>

#include "gst-nvquery.h"
#include "gstnvdsmeta.h"

#include "gst-nvevent.h"
#include "gst-nvmessage.h"

#include "gstnvdsaudiotemplate_meta.h"
#include "nvdscustomlib_base.h"

#include "nvbufaudio.h"
#include "gst_nvdsaudio.h"

#define CUSTOM_METADATA_TYPE 0x100
#define CUSTOM_METADATA_SIZE 64

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string(NVDS_META_STRING);

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

class SampleAlgorithm : public DSCustomLibraryBase {
 public:
  SampleAlgorithm() {
    m_vectorProperty.clear();
    outputthread_stopped = false;
  }

  /* Set Init Parameters */
  virtual bool SetInitParams(DSCustom_CreateParams *params);

  /* Set Custom Properties  of the library */
  virtual bool SetProperty(Property &prop);

  /* Pass GST events to the library */
  virtual bool HandleEvent(GstEvent *event);

  virtual char* QueryProperties ();

  /* Process Incoming Buffer */
  virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

  /* Retrun Compatible Caps */
  virtual GstCaps *GetCompatibleCaps(GstPadDirection direction,
                                     GstCaps *in_caps, GstCaps *othercaps);

  /* Deinit members */
  ~SampleAlgorithm();

 private:
  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

  int doWork(GstBuffer *inbuf, GstBuffer *outbuf);

 public:
  guint source_id = 0;
  guint m_frameNum = 0;
  guint m_noiseFactor = 0;
  bool m_transformMode = false;
  bool hw_caps = false;
  gboolean m_stop = false;

  /* Queue and Lock Management */
  std::queue<PacketInfo> m_processQ;
  std::mutex m_processLock;
  std::condition_variable m_processCV;

  /* Vector Containing Key:Value Pair of Custom Lib Properties */
  std::vector<Property> m_vectorProperty;

  bool outputthread_stopped = false;

  /* Output Thread Pointer */
  std::thread *m_outputThread = NULL;

  GstBufferPool *m_dsBufferPool = NULL;
};

// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(GObject *params);
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(
    GObject *params) {
  return new SampleAlgorithm();
}

// Set Init Parameters
bool SampleAlgorithm::SetInitParams(DSCustom_CreateParams *params) {
  DSCustomLibraryBase::SetInitParams(params);

  if (hw_caps == true)
  {
      GstStructure *structure;
      const gchar* format;
      GstAllocationParams allocation_params;
      GstNvDsAudioAllocatorParams allocator_params = { 0 };

      memset (&allocator_params, 0, sizeof(GstNvDsAudioAllocatorParams));

      structure = gst_caps_get_structure (params->m_outCaps, 0);

      if (!gst_structure_get_int (structure, "rate",
                  (gint *) &allocator_params.rate))
          return false;
      if (!gst_structure_get_int (structure, "channels",
                  (gint *) &allocator_params.channels))
          return false;
      format = gst_structure_get_string (structure, "format");
      if (!strcmp(format, "S16LE"))
          allocator_params.format = NVBUF_AUDIO_S16LE;
      else if (!strcmp (format, "F32LE"))
          allocator_params.format = NVBUF_AUDIO_F32LE;
      else
          return false;

      allocator_params.batchSize        = params->m_batchSize;
      allocator_params.isContiguous     = true;
#if defined(__aarch64__)
      allocator_params.memType          = NVDS_MEM_SYSTEM;
#else
      allocator_params.gpuId            = params->m_gpuId;
      allocator_params.memType          = NVDS_MEM_CUDA_PINNED;
#endif
      allocator_params.layout           = NVBUF_AUDIO_INTERLEAVED;
      allocator_params.bpf              = 4;
      allocator_params.bufferLength     = 441000;


      m_dsBufferPool = gst_buffer_pool_new ();

      GstStructure *config = gst_buffer_pool_get_config (m_dsBufferPool);
      gst_buffer_pool_config_set_params (config, nullptr,
              sizeof (GstNvDsAudioMemory), 3, 3);

      GstAllocator *allocator = gst_nvdsaudio_allocator_new (&allocator_params);

      memset (&allocation_params, 0, sizeof (allocation_params));
      gst_buffer_pool_config_set_allocator (config, allocator,
              &allocation_params);

      if (!gst_buffer_pool_set_config (m_dsBufferPool, config)) {
          g_object_unref (m_dsBufferPool);
          return false;
      }

      if (!gst_buffer_pool_set_active (m_dsBufferPool, TRUE)) {
          return false;
      }
  }

  m_outputThread = new std::thread(&SampleAlgorithm::OutputThread, this);

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps *SampleAlgorithm::GetCompatibleCaps(GstPadDirection direction,
                                            GstCaps *in_caps,
                                            GstCaps *othercaps) {
  GstCapsFeatures *feature = NULL;
  feature = gst_caps_get_features(in_caps, 0);
  if (gst_caps_features_contains(feature, "memory:NVMM")) {
    hw_caps = true;
  }

  return othercaps;
}

char *SampleAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "CUSTOM LIBRARY PROPERTIES\n \t\t\tcustomlib-props=\"noise-factor:x\" x = { 0 <= x < 255 } ");
    return str;
}

bool SampleAlgorithm::HandleEvent(GstEvent *event) {
  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_EOS:
      m_processLock.lock();
      m_stop = true;
      m_processCV.notify_all();
      m_processLock.unlock();
      while (outputthread_stopped == false) {
        // g_print ("waiting for processq to be empty, buffers in processq =
        // %ld\n", m_processQ.size());
        g_usleep(1000);
      }
      break;
    default:
      break;
  }

  if (GST_EVENT_TYPE(event) == GST_EVENT_SINK_MESSAGE) {
    GstMessage *msg = NULL;
    //g_print("Sink message event on nvdsaudiotemplate - custom lib.\n");
    gst_event_parse_sink_message (event, &msg);
    gst_nvmessage_parse_stream_eos(msg, &source_id);
  }

  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED) {
    gst_nvevent_parse_pad_added(event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED) {
    gst_nvevent_parse_pad_deleted(event, &source_id);
  }
  return true;
}

// Set Custom Library Specific Properties
bool SampleAlgorithm::SetProperty(Property &prop) {
  std::cout << "Inside Custom Lib : Setting Prop Key=" << prop.key
            << " Value=" << prop.value << std::endl;
  m_vectorProperty.emplace_back(prop.key, prop.value);

  if (prop.key.compare("noise-factor") == 0) {
    try {
      m_noiseFactor = stof(prop.value);
      if (m_noiseFactor > 255 || m_noiseFactor < 0) {
        throw std::out_of_range("out of range noise factor");
      }
    } catch (std::out_of_range &e) {
      std::cout << "Out of Range Noise Factor, provide between 0 and 255"
                << std::endl;
      return false;
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << std::endl;
      return false;
    } catch (...) {
      std::cout << "caught exception" << std::endl;
      return false;
    }
  }

  return true;
}

/* Deinitialize the Custom Lib context */
SampleAlgorithm::~SampleAlgorithm() {
  std::unique_lock<std::mutex> lk(m_processLock);
  // std::cout << "Process Q Empty : " << m_processQ.empty() << std::endl;
  m_processCV.wait(lk, [&] { return m_processQ.empty(); });
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }
}

int SampleAlgorithm::doWork(GstBuffer *inbuf, GstBuffer *outbuf) {
  GstMapInfo in_map_info = {0};
  GstMapInfo out_map_info = {0};

  memset(&in_map_info, 0, sizeof(in_map_info));
  if (hw_caps == true)
      memset(&out_map_info, 0, sizeof(out_map_info));

  if ((m_noiseFactor != 0)  ||
          ((m_noiseFactor == 0) && (hw_caps == true)))
  {
    /* Map the buffer contents and get the pointer to NvBufSurface. */
    if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
      GST_ELEMENT_ERROR(
          m_element, STREAM, FAILED,
          ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__),
          (NULL));
      return -1;
    }

    if (hw_caps == true)
    {
        if (!gst_buffer_map(outbuf, &out_map_info, GST_MAP_WRITE)) {
            GST_ELEMENT_ERROR(
                    m_element, STREAM, FAILED,
                    ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__),
                    (NULL));
            gst_buffer_unmap (inbuf, &in_map_info);
            return -1;
        }
    }

    /*(hw_caps == false) ? std::cout << "SW CAPS audio/x-raw" << std::endl :
      std::cout << "HW CAPS audio/x-raw(memory:NVMM)" << std::endl;*/
    if (hw_caps == false) {
      /*
       * nvdsaudiotemplate plugin is connected before nvstreammux and operates
       * on SW buffers i.e. audio/x-raw audiosrc -> nvdsaudiotemplate ->
       * nvstreammux -> ...
       * */
      guint8 *data = (guint8 *)in_map_info.data;
      unsigned int i = 0;
      for (i = 0; i < in_map_info.size; i++) {
        data[i] = data[i] + m_noiseFactor;
      }
    } else {
      /*
       * nvdsaudiotemplate plugin is connected after nvstreammux and operates on
       * HW buffers i.e. audio/x-raw(memory:NVMM) audiosrc -> nvstreammux ->
       * nvdsaudiotemplate ...
       * */
      NvBufAudio *src_audio_batch = (NvBufAudio *)in_map_info.data;
      NvBufAudio *dst_audio_batch = (NvBufAudio *)out_map_info.data;
      dst_audio_batch->numFilled = src_audio_batch->numFilled;
      //NvBufAudio *dst_audio_batch = (NvBufAudio *)memory->batch;
      for (guint index = 0; index < src_audio_batch->numFilled; index++)
      {
        dst_audio_batch->audioBuffers[index].bufPts =
              src_audio_batch->audioBuffers[index].bufPts;
        dst_audio_batch->audioBuffers[index].duration =
              src_audio_batch->audioBuffers[index].duration;

        guint8 *data = (guint8 *)src_audio_batch->audioBuffers[index].dataPtr;
        guint8 *odata = (guint8 *)dst_audio_batch->audioBuffers[index].dataPtr;
        dst_audio_batch->audioBuffers[index].dataSize =
            src_audio_batch->audioBuffers[index].dataSize;
        dst_audio_batch->audioBuffers[index].sourceId = index;
        for (guint i = 0; i < src_audio_batch->audioBuffers[index].dataSize;
             i++)
        {
          odata[i] = data[i] + m_noiseFactor;
        }
      }
    }

    gst_buffer_unmap(inbuf, &in_map_info);
    if (hw_caps == true)
        gst_buffer_unmap(outbuf, &out_map_info);
  }
  return 0;
}

/* Process Buffer */
BufferResult SampleAlgorithm::ProcessBuffer(GstBuffer *inbuf) {
  m_frameNum++;
  // g_print("CustomLib: ---> Inside %s frame_num = %d\n", __func__,
  // m_frameNum);

  /*
   * If its not multi threaded processing call doWork here
   * and remove explicit thread creation code
   * */
  /*
  int result = doWork (inbuf);
  if (result == -1)
      return BufferResult::Buffer_Error;
  */

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;

  if (hw_caps == false) {
    /* Mechanism to attach custom metadata to GstBuf which can be retrieved in
     * the downstream plugin */
    GstAudioTemplateMeta *audio_template_meta =
        (GstAudioTemplateMeta *)gst_buffer_add_meta(
            inbuf, GST_AUDIO_TEMPLATE_META_INFO, NULL);

    audio_template_meta->frame_count = m_frameNum;
    audio_template_meta->custom_metadata_type = CUSTOM_METADATA_TYPE;
    audio_template_meta->custom_metadata_size = CUSTOM_METADATA_SIZE;
    audio_template_meta->custom_metadata_ptr =
        calloc(CUSTOM_METADATA_SIZE, sizeof(char));
    strcpy((char *)audio_template_meta->custom_metadata_ptr, "custom_meta_str");
  }

  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

#if 0
  /* To check attached custom metadata by retrieving through gst_buffer_get_meta API */
  GstAudioTemplateMeta *meta =
      (GstAudioTemplateMeta *) gst_buffer_get_meta (inbuf, GST_AUDIO_TEMPLATE_META_API_TYPE);
  if (!meta)
  {
      g_print ("NO META RETRIEVED\n");
  }
  else
  {
      g_print ("frame_number = %d string = %s ptr = %p\n",
              meta->frame_count, (char *)meta->custom_metadata_ptr, meta->custom_metadata_ptr);
  }
#endif
  /*
   * outputthread gets scheduled less frequently than the thread in which
   * submit_input_buffer is called, this leads increasing gap between
   * gst_pad_push to downstream component and buffer queued, to get around this
   * below sleep is added to slow down this thread
   * TODO check the gap and wait to maintain the threshold
   * */
  g_usleep(1);

  return BufferResult::Buffer_Async;
  // return BufferResult::Buffer_Ok; // in case of non-threaded implementation
}

/* Output Processing Thread */
void SampleAlgorithm::OutputThread(void) {
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  std::unique_lock<std::mutex> lk(m_processLock);

  /* Run till signalled to stop. */
  while (1) {
    /* Wait if processing queue is empty. */
    if (m_processQ.empty()) {
      if (m_stop == true) {
        break;
      }
      m_processCV.wait(lk);
      continue;
    }

    PacketInfo packetInfo = m_processQ.front();
    m_processQ.pop();

    m_processCV.notify_all();
    lk.unlock();

    // Add custom algorithm logic here
    // Once buffer processing is done, push the buffer to the downstream by
    // using gst_pad_push function

    if (hw_caps == true)
    {
        flow_ret =
            gst_buffer_pool_acquire_buffer (m_dsBufferPool, &outBuffer, nullptr);
        if (flow_ret != GST_FLOW_OK)
        {
            printf("Failed to acquire buffer FLOW RET = %d\n", flow_ret);
            return;
        }

        if (!gst_buffer_copy_into (outBuffer,
                    packetInfo.inbuf, GST_BUFFER_COPY_META, 0, -1)) {
            GST_DEBUG_OBJECT (m_element, "Buffer metadata copy failed \n");
        }

        nvds_set_input_system_timestamp(outBuffer, GST_ELEMENT_NAME(m_element));
    }

    int result = doWork(packetInfo.inbuf, outBuffer);
    if (result == -1) {
      std::cout << "DoWork Failed" << std::endl;
    }

    if (hw_caps == false)
    {
        outBuffer = packetInfo.inbuf;
    }

    if (hw_caps == true)
      nvds_set_output_system_timestamp(outBuffer, GST_ELEMENT_NAME(m_element));

    flow_ret = gst_pad_push(GST_BASE_TRANSFORM_SRC_PAD(m_element), outBuffer);
    GST_DEBUG("FLOW RET = %d\n", flow_ret);

    if (hw_caps == true)
    {
      gst_buffer_unref (packetInfo.inbuf);
    }

    lk.lock();
    continue;
  }
  // g_print ("outputthread_stopped setting to TRUE\n");
  outputthread_stopped = true;
  lk.unlock();
  return;
}
