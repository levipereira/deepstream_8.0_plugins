/*
 * Example: Accessing Camera Metadata Downstream
 * 
 * This example demonstrates how to access camera_id and source_id
 * in a probe function placed after nvdsanalytics plugin.
 * 
 * Compile with:
 * g++ -shared -fPIC -o example-metadata-access.so example-metadata-access.cpp \
 *     `pkg-config --cflags --libs gstreamer-1.0` \
 *     -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *     -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta
 */

#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

// Camera metadata structure (must match the one in gst-nvmultiurisrcbincreator.cpp)
#define NVDS_GST_META_CAMERA_INFO (nvds_get_user_meta_type((char *)"NVIDIA.NVDS_GST_META_CAMERA_INFO"))

typedef struct _CameraInfoMeta
{
  gchar *camera_id;      // Camera identifier from REST API
  guint source_id;       // Source identifier assigned by DeepStream
  gchar *camera_name;    // Camera name from REST API
  gchar *camera_url;     // Camera URL/URI from REST API
} CameraInfoMeta;

/**
 * Probe function to access camera metadata downstream after nvdsanalytics
 * 
 * This function demonstrates both methods of accessing camera information:
 * 1. Standard sensorInfo metadata (existing functionality)
 * 2. Enhanced camera metadata via NvDsUserMeta (new functionality)
 */
static GstPadProbeReturn
camera_metadata_probe_func (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  if (!batch_meta) {
    return GST_PAD_PROBE_OK;
  }

  g_print("=== Processing Batch (Frame Count: %u) ===\n", batch_meta->num_frames_in_batch);

  // Iterate through all frames in the batch
  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
       l_frame != NULL; l_frame = l_frame->next) {
    
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    
    g_print("\n--- Frame %u ---\n", frame_meta->frame_num);
    
    // Method 1: Access standard sensorInfo metadata (existing functionality)
    g_print("Standard Metadata:\n");
    g_print("  Source ID: %u\n", frame_meta->sensorInfo_meta.source_id);
    g_print("  Sensor ID: %s\n", frame_meta->sensorInfo_meta.sensor_id ? 
             frame_meta->sensorInfo_meta.sensor_id : "N/A");
    g_print("  Sensor Name: %s\n", frame_meta->sensorInfo_meta.sensor_name ? 
             frame_meta->sensorInfo_meta.sensor_name : "N/A");
    g_print("  URI: %s\n", frame_meta->sensorInfo_meta.uri ? 
             frame_meta->sensorInfo_meta.uri : "N/A");
    
    // Method 2: Access enhanced camera metadata via NvDsUserMeta (new functionality)
    g_print("Enhanced Camera Metadata:\n");
    
    gboolean camera_meta_found = FALSE;
    NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
    
    while (l_user_meta != NULL) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
      
      if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
        CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
        
        g_print("  Camera ID: %s\n", camera_meta->camera_id ? camera_meta->camera_id : "N/A");
        g_print("  Source ID: %u\n", camera_meta->source_id);
        g_print("  Camera Name: %s\n", camera_meta->camera_name ? camera_meta->camera_name : "N/A");
        g_print("  Camera URL: %s\n", camera_meta->camera_url ? camera_meta->camera_url : "N/A");
        
        camera_meta_found = TRUE;
        break;
      }
      
      l_user_meta = l_user_meta->next;
    }
    
    if (!camera_meta_found) {
      g_print("  No enhanced camera metadata found\n");
    }
    
    // Example: Use camera metadata for processing decisions
    if (camera_meta_found) {
      // You can now use the camera_id and source_id for:
      // - Routing frames to different processing paths
      // - Logging with camera identification
      // - Database operations with camera context
      // - Analytics per camera
      
      g_print("  Processing frame from camera: %s\n", 
               frame_meta->sensorInfo_meta.sensor_id ? 
               frame_meta->sensorInfo_meta.sensor_id : "unknown");
    }
    
    // Access object metadata if needed
    if (frame_meta->num_obj_meta > 0) {
      g_print("  Objects detected: %u\n", frame_meta->num_obj_meta);
      
      for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; 
           l_obj != NULL; l_obj = l_obj->next) {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
        
        g_print("    Object %u: class_id=%u, confidence=%.2f, camera=%s\n",
                obj_meta->object_id, obj_meta->class_id, obj_meta->confidence,
                frame_meta->sensorInfo_meta.sensor_id ? 
                frame_meta->sensorInfo_meta.sensor_id : "unknown");
      }
    }
  }

  return GST_PAD_PROBE_OK;
}

/**
 * Helper function to get camera_id from frame metadata
 */
gchar* get_camera_id_from_frame_meta(NvDsFrameMeta *frame_meta)
{
  if (!frame_meta) {
    return NULL;
  }
  
  // First try enhanced metadata
  NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
  while (l_user_meta != NULL) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
    
    if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
      CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
      return camera_meta->camera_id;
    }
    
    l_user_meta = l_user_meta->next;
  }
  
  // Fallback to standard metadata
  return (gchar*)frame_meta->sensorInfo_meta.sensor_id;
}

/**
 * Helper function to get source_id from frame metadata
 */
guint get_source_id_from_frame_meta(NvDsFrameMeta *frame_meta)
{
  if (!frame_meta) {
    return 0;
  }
  
  // First try enhanced metadata
  NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
  while (l_user_meta != NULL) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
    
    if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
      CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
      return camera_meta->source_id;
    }
    
    l_user_meta = l_user_meta->next;
  }
  
  // Fallback to standard metadata
  return frame_meta->sensorInfo_meta.source_id;
}

/**
 * Example of how to add the probe to a pipeline element
 * Call this function after creating your pipeline and before setting it to PLAYING state
 */
void add_camera_metadata_probe(GstElement *element, const gchar *pad_name)
{
  GstPad *pad = gst_element_get_static_pad(element, pad_name);
  if (!pad) {
    g_print("Failed to get pad %s from element %s\n", pad_name, GST_ELEMENT_NAME(element));
    return;
  }
  
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, 
                    camera_metadata_probe_func, NULL, NULL);
  
  g_print("Added camera metadata probe to %s:%s\n", 
          GST_ELEMENT_NAME(element), pad_name);
  
  gst_object_unref(pad);
}

/**
 * Example pipeline setup with camera metadata probe
 * 
 * This shows where to place the probe in a typical DeepStream pipeline:
 * nvmultiurisrcbin -> nvinfer -> nvtracker -> nvdsanalytics -> queue -> [PROBE HERE]
 */
void setup_example_pipeline_with_probe()
{
  // This is pseudo-code to show probe placement
  /*
  GstElement *pipeline = gst_pipeline_new("camera-metadata-pipeline");
  GstElement *nvmultiurisrcbin = gst_element_factory_make("nvmultiurisrcbin", "source");
  GstElement *nvinfer = gst_element_factory_make("nvinfer", "infer");
  GstElement *nvtracker = gst_element_factory_make("nvtracker", "tracker");
  GstElement *nvdsanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
  GstElement *queue = gst_element_factory_make("queue", "queue");
  GstElement *sink = gst_element_factory_make("fakesink", "sink");
  
  // Add elements to pipeline
  gst_bin_add_many(GST_BIN(pipeline), nvmultiurisrcbin, nvinfer, nvtracker, 
                   nvdsanalytics, queue, sink, NULL);
  
  // Link elements
  gst_element_link_many(nvmultiurisrcbin, nvinfer, nvtracker, 
                        nvdsanalytics, queue, sink, NULL);
  
  // Add probe AFTER nvdsanalytics (on queue's sink pad or any downstream element)
  add_camera_metadata_probe(queue, "sink");
  
  // Set pipeline to playing
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  */
}

/*
 * Usage in your application:
 * 
 * 1. Include this code in your DeepStream application
 * 2. Add the probe after nvdsanalytics plugin
 * 3. Access camera_id and source_id in your processing logic
 * 4. Use the helper functions for easy metadata access
 * 
 * Example REST API calls to test:
 * 
 * curl -X POST http://localhost:9000/api/v1/streams \
 *   -H "Content-Type: application/json" \
 *   -d '{
 *     "camera_id": "camera_001",
 *     "camera_url": "rtsp://192.168.1.100:554/stream",
 *     "camera_name": "Main Entrance"
 *   }'
 * 
 * curl -X POST http://localhost:9000/api/v1/streams \
 *   -H "Content-Type: application/json" \
 *   -d '{
 *     "camera_id": "camera_002", 
 *     "camera_url": "rtsp://192.168.1.101:554/stream",
 *     "camera_name": "Side Door"
 *   }'
 */
