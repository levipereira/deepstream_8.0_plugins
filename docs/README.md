# DeepStream Camera Metadata Implementation

## Quick Start

This implementation enables `camera_id` and `source_id` to be available downstream after the `nvdsanalytics` plugin in DeepStream 8.0.

### What's New

- ✅ **Enhanced Metadata Support**: Camera information persists through entire pipeline
- ✅ **REST API Integration**: Dynamic source addition with metadata
- ✅ **Dual Access Methods**: Both standard and enhanced metadata access
- ✅ **Zero Configuration**: Automatic metadata attachment
- ✅ **Memory Efficient**: Uses NVIDIA metadata pools

### Key Features

1. **Automatic Metadata Attachment**: Camera information is automatically attached to every frame
2. **Downstream Availability**: Access `camera_id` and `source_id` after `nvdsanalytics`
3. **REST API Integration**: Add/remove sources dynamically with metadata
4. **Backward Compatible**: Existing sensorInfo metadata still works
5. **Performance Optimized**: Minimal overhead on pipeline performance

## Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `libs/gstnvdscustomhelper/gst-nvmultiurisrcbincreator.cpp` | Metadata Management | Added CameraInfoMeta structure and attachment functions |
| `gst-plugins/gst-nvmultiurisrcbin/gstdsnvmultiurisrcbin.cpp` | REST API Integration | Added metadata setup on source addition |

## Usage

### 1. Add Source via REST API

```bash
curl -X POST http://localhost:9000/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "camera_001",
    "camera_url": "rtsp://192.168.1.100:554/stream",
    "camera_name": "Main Entrance"
  }'
```

### 2. Access Metadata Downstream

```cpp
// In your probe function after nvdsanalytics
NvDsMetaList *l_user_meta = frame_meta->frame_user_meta_list;
while (l_user_meta != NULL) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user_meta->data);
  
  if (user_meta->base_meta.meta_type == NVDS_GST_META_CAMERA_INFO) {
    CameraInfoMeta *camera_meta = (CameraInfoMeta *)user_meta->user_meta_data;
    
    printf("Camera: %s, Source: %u\n", 
           camera_meta->camera_id, camera_meta->source_id);
  }
  
  l_user_meta = l_user_meta->next;
}
```

### 3. Build Components

```bash
# Build nvmultiurisrcbin
cd sources/gst-plugins/gst-nvmultiurisrcbin
sudo make clean && sudo make

# Build creator library
cd sources/libs/gstnvdscustomhelper
sudo make clean && sudo make
```

## Documentation

- **[Implementation Guide](camera-metadata-implementation.md)** - Complete technical documentation
- **[Example Code](example-metadata-access.cpp)** - Working code examples
- **[Original 7.1 Implementation](../implmentacao.md)** - Reference implementation

## Pipeline Architecture

```
REST API → nvmultiurisrcbin → gst-infer → tracker → nvdsanalytics → queue → YOUR_PROBE
           ↓
       Metadata attached automatically and available throughout pipeline
```

## Testing

### Functional Test
1. Start your DeepStream application with nvmultiurisrcbin
2. Add a source via REST API
3. Verify camera metadata is available in your downstream probe
4. Check logs for metadata setup confirmation

### Expected Output
```
Camera metadata setup: camera_id=camera_001, source_id=0, camera_name=Main Entrance, camera_url=rtsp://...
Enhanced metadata - Camera: camera_001, Source: 0, Name: Main Entrance, URL: rtsp://...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Metadata not found | Ensure probe is placed after nvdsanalytics |
| Build errors | Check include paths and library dependencies |
| Memory issues | Verify proper cleanup in release functions |
| REST API fails | Check nvmultiurisrcbin configuration and ports |

## Version Compatibility

- **DeepStream**: 8.0+
- **Platform**: x86_64, Jetson
- **GStreamer**: 1.0+
- **CUDA**: Any version supported by DeepStream 8.0

## Support

For issues and questions:
1. Check the [Implementation Guide](camera-metadata-implementation.md)
2. Review the [Example Code](example-metadata-access.cpp)
3. Verify your pipeline configuration matches the expected architecture

---

**Status**: ✅ Implemented and Tested  
**Version**: 1.0  
**DeepStream**: 8.0+
