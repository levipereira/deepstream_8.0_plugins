# Camera Info UserMeta Implementation

## Overview

This document describes the implementation of camera metadata attachment using `NvDsUserMeta` in the DeepStream `nvmultiurisrcbin` plugin. The implementation allows `camera_id` and `camera_name` to be available downstream after the `nvdsanalytics` plugin, following the official NVIDIA pattern from `deepstream_user_metadata_app.c`.

## Architecture

```
Pipeline: multiurisrcbin > gst-infer > tracker > nvdsanalytics > queue
                                                      ↓
                                              Camera metadata
                                              (NvDsUserMeta)
                                                      ↓
                                              Downstream access
```

## Implementation Details

### 1. Custom Metadata Structure

**File**: `libs/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp`

```cpp
/**
 * @brief Camera metadata structure for NvDsUserMeta attachment
 * 
 * This structure contains camera identification information that is attached
 * to each frame as NvDsUserMeta and can be accessed downstream via standard DeepStream APIs.
 * Fixed-size arrays are used for efficient memory management and downstream compatibility.
 */
typedef struct _CameraInfoMeta
{
  gchar camera_id[64];       ///< Camera identifier from REST API
  guint source_id;           ///< Source identifier assigned by DeepStream  
  gchar camera_name[64];     ///< Camera name from REST API
  gchar camera_url[128];     ///< Camera URL/URI from REST API
} CameraInfoMeta;
```

### 2. Metadata Type Registration

```cpp
#define NVDS_GST_META_CAMERA_INFO (nvds_get_user_meta_type((char *)"NVIDIA.NVDS_GST_META_CAMERA_INFO"))
```

### 3. Copy and Release Functions

Following the official NVIDIA pattern from `deepstream_user_metadata_app.c`:

#### Copy Function
```cpp
static gpointer camera_meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  CameraInfoMeta *src_meta = (CameraInfoMeta *) user_meta->user_meta_data;
  CameraInfoMeta *dst_meta = NULL;

  if (!src_meta) {
    GST_DEBUG ("camera_meta_copy_func called with NULL src_meta");
    return NULL;
  }

  /* Allocate new structure following NVIDIA pattern */
  dst_meta = (CameraInfoMeta *) g_malloc0 (sizeof (CameraInfoMeta));
  if (!dst_meta) {
    GST_ERROR ("Failed to allocate memory for camera metadata copy");
    return NULL;
  }

  /* Copy the entire structure safely using NVIDIA memcpy pattern */
  memcpy(dst_meta, src_meta, sizeof(CameraInfoMeta));

  GST_DEBUG ("camera_meta_copy_func completed - source_id: %u", dst_meta->source_id);
  return (gpointer) dst_meta;
}
```

#### Release Function
```cpp
static void camera_meta_release_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;

  if (!user_meta) {
    GST_DEBUG ("camera_meta_release_func called with NULL user_meta");
    return;
  }

  /* NVIDIA official pattern from deepstream_user_metadata_app.c */
  if (user_meta->user_meta_data) {
    GST_DEBUG ("camera_meta_release_func releasing data - source_id: %u", 
               ((CameraInfoMeta*)user_meta->user_meta_data)->source_id);
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;  /* Critical: prevent double free */
  }

  GST_DEBUG ("camera_meta_release_func completed");
}
```

### 4. Metadata Attachment Function

```cpp
static void attach_camera_metadata_to_frame (NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                                           const gchar *camera_id, guint source_id,
                                           const gchar *camera_name, const gchar *camera_url)
{
  NvDsUserMeta *user_meta = NULL;
  CameraInfoMeta *camera_meta = NULL;
  NvDsMetaType user_meta_type = NVDS_GST_META_CAMERA_INFO;

  /* Validate inputs */
  if (!batch_meta || !frame_meta) {
    GST_ERROR ("Invalid batch_meta or frame_meta");
    return;
  }

  /* Acquire NvDsUserMeta from pool - NVIDIA official pattern */
  user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
  if (!user_meta) {
    GST_ERROR ("Failed to acquire user meta from pool");
    return;
  }

  /* Create camera metadata structure - following NVIDIA g_malloc0 pattern */
  camera_meta = (CameraInfoMeta *) g_malloc0 (sizeof (CameraInfoMeta));
  if (!camera_meta) {
    GST_ERROR ("Failed to allocate camera metadata");
    return;
  }

  /* Populate camera metadata with safe string copying */
  g_strlcpy(camera_meta->camera_id, camera_id ? camera_id : "", sizeof(camera_meta->camera_id));
  camera_meta->source_id = source_id;
  g_strlcpy(camera_meta->camera_name, camera_name ? camera_name : "", sizeof(camera_meta->camera_name));
  g_strlcpy(camera_meta->camera_url, camera_url ? camera_url : "", sizeof(camera_meta->camera_url));

  /* Set NvDsUserMeta following NVIDIA official pattern */
  user_meta->user_meta_data = (void *) camera_meta;
  user_meta->base_meta.meta_type = user_meta_type;
  user_meta->base_meta.copy_func = (NvDsMetaCopyFunc) camera_meta_copy_func;
  user_meta->base_meta.release_func = (NvDsMetaReleaseFunc) camera_meta_release_func;

  /* Add NvDsUserMeta to frame level */
  nvds_add_user_meta_to_frame(frame_meta, user_meta);

  GST_DEBUG ("Successfully attached camera metadata - camera_id: %s, source_id: %u",
             camera_meta->camera_id, camera_meta->source_id);
}
```

### 5. Probe Function Integration

The metadata attachment is integrated into the existing probe function:

```cpp
static GstPadProbeReturn s_nvmultiurisrcbincreator_probe_func_add_sensorInfo (GstPad * pad,
    GstPadProbeInfo * info, gpointer u_data)
{
  // ... existing code ...

  /* Camera metadata attachment using NvDsUserMeta - NVIDIA official approach */
  if (srcInfo && srcInfo->config) {
      GST_LOG ("Attaching camera metadata - source_id=%u, camera_id=%s, camera_name=%s", 
               srcInfo->config->source_id, 
               srcInfo->config->sensorId ? srcInfo->config->sensorId : "NULL",
               srcInfo->config->sensorName ? srcInfo->config->sensorName : "NULL");
      
      /* Use official NVIDIA NvDsUserMeta approach for metadata attachment */
      attach_camera_metadata_to_frame(batch_meta, frame_meta,
                                      srcInfo->config->sensorId,     /* camera_id */
                                      srcInfo->config->source_id,    /* source_id */
                                      srcInfo->config->sensorName,   /* camera_name */
                                      srcInfo->config->uri);         /* camera_url */
      
      GST_DEBUG ("Camera metadata attached successfully - camera_id='%s', camera_name='%s'",
                 srcInfo->config->sensorId ? srcInfo->config->sensorId : "NULL",
                 srcInfo->config->sensorName ? srcInfo->config->sensorName : "NULL");
  }

  return GST_PAD_PROBE_OK;
}
```

## Downstream Access

The camera metadata can be accessed downstream using standard DeepStream APIs:

### C/C++ Access Pattern

```cpp
// Access frame-level user metadata
NvDsMetaList *l_user = frame_meta->frame_user_meta_list;
while (l_user != NULL) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
    
    // Check for our custom camera metadata type
    if (user_meta && user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
        CameraInfoMeta *camera_meta = (CameraInfoMeta *) user_meta->user_meta_data;
        
        if (camera_meta) {
            // Access camera information
            const gchar *camera_id = camera_meta->camera_id;
            const gchar *camera_name = camera_meta->camera_name;
            guint source_id = camera_meta->source_id;
            const gchar *camera_url = camera_meta->camera_url;
            
            // Process camera metadata...
        }
    }
    
    l_user = l_user->next;
}
```

## Key Features

1. **Official NVIDIA Pattern**: Follows the exact pattern from `deepstream_user_metadata_app.c`
2. **Memory Safety**: Uses fixed-size arrays and proper memory management
3. **Thread Safety**: Leverages DeepStream's built-in metadata management
4. **REST API Compatible**: Works with dynamic source addition/removal
5. **Downstream Access**: Metadata available after `nvdsanalytics` plugin
6. **Standard DeepStream APIs**: Uses official DeepStream metadata access patterns

## Memory Management

- **Allocation**: Uses `g_malloc0()` for consistent memory allocation
- **Copying**: Uses `memcpy()` for efficient structure copying
- **Release**: Proper cleanup with `g_free()` and NULL pointer setting
- **Pool Management**: Leverages DeepStream's metadata pool system

## Debugging

The implementation includes comprehensive debug logging:

- `GST_DEBUG`: Detailed operation logging
- `GST_LOG`: Metadata attachment confirmation
- `GST_ERROR`: Error condition reporting

Enable debug logging with:
```bash
export GST_DEBUG=3
```

## Files Modified

1. **`libs/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp`**
   - Added `CameraInfoMeta` structure definition
   - Added `camera_meta_copy_func()` function
   - Added `camera_meta_release_func()` function
   - Added `attach_camera_metadata_to_frame()` function
   - Modified `s_nvmultiurisrcbincreator_probe_func_add_sensorInfo()` probe function
   - Added metadata type registration `NVDS_GST_META_CAMERA_INFO`

## Testing

The implementation has been tested with:
- Dynamic source addition via REST API
- Multiple concurrent sources
- Memory leak detection
- Downstream plugin compatibility
- Metadata persistence through pipeline

## Compatibility

- **DeepStream Version**: 8.0
- **GStreamer**: 1.18+
- **C/C++**: Standard C99/C++11 compatible
- **Memory Management**: GLib 2.0+ compatible
