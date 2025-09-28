# DeepStream 8.0 Plugins

This repository contains the official source code for DeepStream 8.0 libraries and plugins, providing enhanced functionality and custom implementations for NVIDIA's DeepStream SDK.

## Overview

DeepStream 8.0 Plugins extends the capabilities of NVIDIA's DeepStream platform with custom implementations and additional features. This repository serves as the primary source for official DeepStream 8.0 library components and custom plugins.

## Repository Structure

This repository is organized with feature-specific branches, each containing targeted implementations and enhancements:

### Active Branches

#### [`feature/cameraInfo-UserMeta-downstream-support`](https://github.com/levipereira/deepstream_8.0_plugins/tree/feature/camera-metadata-storage-global-ctypes)

This branch contains the implementation of camera metadata attachment using `NvDsUserMeta` in the DeepStream `nvmultiurisrcbin` plugin. The implementation enables `camera_id` and `camera_name` to be available downstream after the `nvdsanalytics` plugin, following the official NVIDIA pattern from `deepstream_user_metadata_app.c`.



**Key Features:**
- Camera metadata propagation through the DeepStream pipeline
- Downstream availability of camera identification information
- Official NVIDIA pattern compliance
- Enhanced analytics capabilities with camera context

**Documentation:** Detailed implementation documentation is available in the [camera_info_usermeta_implementation.md](https://github.com/levipereira/deepstream_8.0_plugins/blob/feature/cameraInfo-UserMeta-downstream-support/libs/gstnvdscustomhelper/camera_info_usermeta_implementation.md) file.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/levipereira/deepstream_8.0_plugins.git
   ```

2. Checkout the desired feature branch:
   ```bash
   git checkout feature/cameraInfo-UserMeta-downstream-support
   ```

3. Follow the specific implementation documentation for each feature branch.

## Requirements

- NVIDIA DeepStream SDK 8.0
- CUDA-compatible GPU
- Ubuntu 22.04 LTS  

## Contributing

New features and enhancements are implemented in dedicated feature branches. Each branch contains:
- Complete implementation of the specific feature
- Comprehensive documentation
- Example usage and integration guides
 

## License

This project follows the same licensing terms as the official NVIDIA DeepStream SDK.

## Support

For questions and support regarding specific implementations, please refer to the documentation within each feature branch or create an issue in this repository.
