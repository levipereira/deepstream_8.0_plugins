/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gst-nvevent.h"
#include "gstaudio2video.h"

#include "nvdscustomlib_base.hpp"

enum
{
  STYLE_DOTS = 0,
  STYLE_LINES,
  STYLE_COLOR_DOTS,
  STYLE_COLOR_LINES,
  NUM_STYLES
};

class SampleAlgorithm : public DSCustomLibraryBase
{
public:
  SampleAlgorithm(GObject *obj): DSCustomLibraryBase(GST_IS_ELEMENT(obj) ? GST_ELEMENT(obj) : nullptr) {
    if (!GST_IS_ELEMENT(obj)) {
        GST_WARNING("Object passed to SampleAlgorithm constructor is not (yet) a GstElement.");
    }
    m_vectorProperty.clear();
  }

  /* Set Init Parameters */
  virtual bool SetInitParams(DSCustom_CreateParams *params);

  /* Set Custom Properties  of the library */
  virtual bool SetProperty(Property &prop);

  virtual bool HandleEvent(GstEvent *event);

  virtual char *QueryProperties ();

  virtual BufferResult ProcessBuffer(GstAudio2Video * baseclass, GstBuffer * audio,
    GstVideoFrame * video);

  /* Deinit members */
  ~SampleAlgorithm();

  guint source_id = 0;
  guint m_style;

  /* filter specific data */
  static void render_sample_nv12(GstAudio2Video * baseclass, guint8 * video, gint16 * audio,
    guint num_samples, guint pitch);

  void (*process) (GstAudio2Video *, guint8 *, gint16 *, guint, guint);

  /* Queue and Lock Management */
  std::mutex m_processLock;
  std::condition_variable m_processCV;

  /* Vector Containing Key:Value Pair of Custom Lib Properties */
  std::vector<Property> m_vectorProperty;
};
gdouble *flt;

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(GObject *obj, DSCustom_CreateParams *params);
// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(GObject *obj, DSCustom_CreateParams *params)
{
    if(G_IS_OBJECT (obj))
	    printf("G_OBJECT(obj) = %p\n", G_OBJECT(obj));

  return new SampleAlgorithm(obj);
}

void SampleAlgorithm::render_sample_nv12(GstAudio2Video *baseclass, guint8 *video,
                                         gint16 *audio, guint num_samples, guint pitch) {
    guint width = GST_VIDEO_INFO_WIDTH(&baseclass->vinfo);
    guint height = GST_VIDEO_INFO_HEIGHT(&baseclass->vinfo);
    guint i,j;
    guint8 *y_plane = (guint8 *)video, *uv_plane = (guint8 *)video + pitch * height;


for (i=1; i<= height; i++)
{
    for (j=0; j<width; j++)
    {
         *(y_plane + j) =  16;
    }
    y_plane =video + (i*pitch);
}

for (i=1; i<= height/2; i++)
{
    for (j=0; j<width; j++)
    {
	 *(uv_plane + j) = 128;
    }
    uv_plane = (video + (pitch * height)) +(i*pitch);
}
    
}

// Set Init Parameters
bool SampleAlgorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);

  g_free (flt);

  flt = g_new0 (gdouble, 36);
  process = render_sample_nv12;

  return true;
}

char *SampleAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "CUSTOM LIBRARY PROPERTIES\n \t\t\tcustomlib-props=\"style:x\" x = { 1 <= x <= 4}");
    return str;
}

bool SampleAlgorithm::HandleEvent (GstEvent *event)
{
  guint source_id = 0;
  g_print("Event Type %s\n", gst_event_type_get_name(event->type));
  switch (GST_EVENT_TYPE(event))
  {
       case GST_EVENT_EOS:
           break;
       default:
           break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED)
  {
      gst_nvevent_parse_pad_added (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
  {
      gst_nvevent_parse_pad_deleted (event, &source_id);
  }
  return true;
}

// Set Custom Library Specific Properties
bool SampleAlgorithm::SetProperty(Property &prop)
{
  std::cout << "Inside Custom Lib : Setting Prop Key=" << prop.key << " Value=" << prop.value << std::endl;
  m_vectorProperty.emplace_back(prop.key, prop.value);

  try
  {
      if (prop.key.compare("style") == 0)
      {
          m_style = stof(prop.value);
          if (m_style > 4)
          {
              throw std::out_of_range ("out of range style");
          }
      }
  }
  catch(std::invalid_argument& e)
  {
      std::cout << "Invalid style" << std::endl;
      return false;
  }
  catch(std::out_of_range& e)
  {
      std::cout << "Out of Range style, provide between 1 and 4" << std::endl;
      return false;
  }

  switch (m_style) {
   case STYLE_DOTS:
      process = render_sample_nv12;
   break;
  }
  return true;
}

/* Deinitialize the Custom Lib context */
SampleAlgorithm::~SampleAlgorithm()
{
    std::unique_lock<std::mutex> lk(m_processLock);
  m_processCV.notify_all();
  if (flt) {
    g_free (flt);
    flt = NULL;
  }
  lk.unlock();

}

/* Process Buffer */
BufferResult SampleAlgorithm::ProcessBuffer (GstAudio2Video * baseclass, GstBuffer * audio,
    GstVideoFrame * video)
{
  GstMapInfo amap;
  guint num_samples;
  GstMapInfo outmap = GST_MAP_INFO_INIT;
  guint height = GST_VIDEO_INFO_HEIGHT (&baseclass->vinfo);
  guint8 *data = NULL;
  guint size=0;
  cudaError_t err ;
  guint8 *video_data = NULL;

  gint channels = 1;

  if (!gst_buffer_map (video->buffer, &outmap, GST_MAP_WRITE))
  {
    GST_ERROR ("output buffer mapinfo failed");
    return BufferResult::Buffer_Error;
  }

  NvBufSurface *op_surf = (NvBufSurface *) outmap.data;
  
  gst_buffer_map (audio, &amap, GST_MAP_READ);

  num_samples = amap.size / (channels * sizeof (gint16));

  size=op_surf->surfaceList[0].pitch * height + (op_surf->surfaceList[0].pitch * height / 2);  ;
  data=(guint8 *)malloc(size);
  video_data = (guint8 *)op_surf->surfaceList[0].dataPtr;

  process (baseclass, data,
      (gint16 *) amap.data, num_samples, op_surf->surfaceList[0].pitch);

  err = cudaMemcpy(video_data, data, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
   g_print("Error: %s\n", cudaGetErrorString(err));
  }

  free(data);
  gst_buffer_unmap (audio, &amap);
  gst_buffer_unmap ((GstBuffer *)video->buffer, &outmap);

  return BufferResult::Buffer_Ok;
}