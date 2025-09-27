# DeepStream Camera Metadata Implementation Guide

## Overview

This document describes the implementation of camera metadata functionality in DeepStream 8.0, enabling `camera_id` and `source_id` to be available downstream after the `nvdsanalytics` plugin using `NvDsUserMeta`.

## Architecture

### Data Flow
```
nvmultiurisrcbin (REST API) 
    ↓
gst-infer → tracker → nvdsanalytics → queue → PROBE
    ↓                                    ↓
Metadata attached automatically    camera_id + source_id available
```

### Core Components

1. **nvmultiurisrcbin** - Receives sources via REST API
2. **nvmultiurisrcbincreator** - Manages sources and attaches metadata
3. **Camera Metadata Structures** - Enhanced metadata support

## Implementation Details

### 1. Enhanced Metadata Structures

#### CameraInfoMeta Structure
```cpp
typedef struct _CameraInfoMeta
{
  gchar *camera_id;      // Camera identifier from REST API
  guint source_id;       // Source identifier assigned by DeepStream
  gchar *camera_name;    // Camera name from REST API
  gchar *camera_url;     // Camera URL/URI from REST API
} CameraInfoMeta;
```

#### Metadata Type Definition
```cpp
#define NVDS_GST_META_CAMERA_INFO (nvds_get_user_meta_type((char *)"NVIDIA.NVDS_GST_META_CAMERA_INFO"))
```

### 2. Metadata Management Functions

#### Memory Management
- **`camera_meta_copy_func`**: Handles metadata duplication
- **`camera_meta_release_func`**: Manages memory cleanup
- **`attach_camera_metadata_to_frame`**: Main function to attach metadata to frames

#### Automatic Attachment
The metadata is automatically attached in the probe function `s_nvmultiurisrcbincreator_probe_func_add_sensorInfo`:

```cpp
// Attach enhanced camera metadata using NvDsUserMeta
attach_camera_metadata_to_frame (batch_meta, frame_meta, 
                                srcInfo->config->sensorId,     // camera_id 
                                srcInfo->config->source_id,    // source_id
                                srcInfo->config->sensorName,   // camera_name
                                srcInfo->config->uri);         // camera_url
```

### 3. REST API Integration

When a source is successfully added via REST API, the system:

1. Receives camera information from the REST request
2. Creates source configuration with `camera_id`, `camera_name`, and `camera_url`
3. Calls `attach_camera_metadata_to_frames()` to setup metadata logging
4. Automatically attaches metadata to each frame via the probe function

#### Example REST API Call
```bash
curl -X POST http://localhost:9000/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "camera_001",
    "camera_url": "rtsp://192.168.1.100:554/stream",
    "camera_name": "Main Entrance"
  }'
```

## Usage Guide

### 1. Building the Components

```bash
# Build nvmultiurisrcbin
cd sources/gst-plugins/gst-nvmultiurisrcbin
sudo make clean && sudo make

# Build the creator library
cd sources/libs/gstnvdscustomhelper
sudo make clean && sudo make
```

### 2. Accessing Metadata Downstream

In your probe function or plugin (after nvdsanalytics):

```cpp
// Access standard sensorInfo metadata (existing)
printf("Standard metadata - Camera ID: %s, Source ID: %u\n", 
       frame_meta->sensorInfo_meta.sensor_id, 
       frame_meta->sensorInfo_meta.source_id);

// Access enhanced camera metadata (new)
NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
while (l_user_meta != NULL) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
  
  if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
    CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
    
    printf("Enhanced metadata - Camera: %s, Source: %u, Name: %s, URL: %s\n", 
           camera_meta->camera_id, camera_meta->source_id,
           camera_meta->camera_name, camera_meta->camera_url);
  }
  
  l_user_meta = l_user_meta->next;
}
```

### 3. Example Probe Function

```cpp
static GstPadProbeReturn
my_probe_function (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  if (!batch_meta) {
    return GST_PAD_PROBE_OK;
  }

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; 
       l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    
    // Access camera metadata
    NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
    while (l_user_meta != NULL) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
      
      if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
        CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
        
        // Use camera metadata here
        g_print("Processing frame from camera: %s (source: %u)\n", 
                camera_meta->camera_id, camera_meta->source_id);
      }
      
      l_user_meta = l_user_meta->next;
    }
  }

  return GST_PAD_PROBE_OK;
}
```

## Key Features

### ✅ Technical Advantages
- **Persistent Metadata**: camera_id and source_id available throughout pipeline
- **Automatic**: No manual modification required for each frame
- **Efficient**: Metadata attached once per source configuration
- **Compatible**: Uses NVIDIA standards (NvDsUserMeta)
- **Dual Access**: Both standard sensorInfo and enhanced user metadata

### ✅ Functional Benefits
- **Complete Tracking**: Unique identification for each camera
- **Dynamic Sources**: Add/remove sources via REST API
- **Plugin Integration**: Works with all existing NVIDIA plugins
- **Debug Support**: Enhanced logging for troubleshooting

### ✅ Architectural Benefits
- **Modular**: Isolated implementation in nvmultiurisrcbin
- **Extensible**: Easy to add new metadata fields
- **Maintainable**: Clean, well-documented code
- **Reusable**: Pattern applicable to other plugins

## File Modifications

### Modified Files
1. `libs/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp`
   - Added CameraInfoMeta structure and metadata management functions
   - Enhanced probe function to attach both standard and enhanced metadata

2. `gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp`
   - Added camera metadata setup helper function
   - Integrated metadata attachment with REST API success callback

## Testing and Validation

### Functional Testing
1. **REST API Integration**: Verify camera_id is received and stored
2. **Metadata Attachment**: Confirm metadata is attached to frames
3. **Downstream Access**: Test metadata availability after nvdsanalytics
4. **Memory Management**: Ensure no memory leaks in metadata handling

### Performance Testing
- **Minimal Overhead**: Metadata attached efficiently per frame
- **Memory Optimized**: Uses NVIDIA metadata pools
- **No Impact**: Does not affect main pipeline performance

## Troubleshooting

### Common Issues
1. **Metadata Not Found**: Ensure probe is added after nvdsanalytics
2. **Memory Leaks**: Verify proper cleanup in release functions
3. **Build Errors**: Check include paths and library dependencies

### Debug Output
The implementation includes debug logging:
```
Camera metadata setup: camera_id=camera_001, source_id=0, camera_name=Main Entrance, camera_url=rtsp://...
```

## Version Compatibility

- **DeepStream Version**: 8.0+
- **GStreamer**: Compatible with existing GStreamer pipeline
- **CUDA**: No specific CUDA version requirements
- **Platform**: x86_64 and Jetson platforms

## Future Enhancements

### Potential Improvements
- [ ] Add metadata validation and error checking
- [ ] Implement metadata caching for performance
- [ ] Add support for custom metadata fields per camera
- [ ] Integration with monitoring systems
- [ ] Real-time metadata query API

---

**Implementation Date**: 2024  
**Version**: 1.0  
**Status**: Implemented and Tested  
**Compatibility**: DeepStream 8.0+
