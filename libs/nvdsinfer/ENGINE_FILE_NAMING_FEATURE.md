# TensorRT Engine File Smart Caching Feature

## Overview

New intelligent engine file naming and auto-discovery system for DeepStream's `nvdsinfer` library. Automatically finds and reuses previously built TensorRT engines, reducing pipeline startup time from minutes to seconds.

---

## Engine File Naming Format

```
{model}_b{batch}_i{width}x{height}_{compute_cap}_{gpu_model}_{trt_version}_{precision}.engine
```

**Examples:**
```
model_b8_i640x640_sm120_rtx5090_trt10.7_fp16.engine
model_b4_i1280x720_sm120_rtx5070_trt10.7_fp16.engine
model_b8_i640x640_dla0_trt10.7_int8.engine
```

### Components

| Component | Description | Example |
|-----------|-------------|---------|
| `{model}` | Model name (from source file) | `model` |
| `b{batch}` | Maximum batch size | `b8` |
| `i{width}x{height}` | Network input dimensions | `i640x640` |
| `{compute_cap}` | GPU compute capability | `sm120` |
| `{gpu_model}` | GPU model name | `rtx5090` |
| `{trt_version}` | TensorRT version | `trt10.7` |
| `{precision}` | Precision mode | `fp16`, `fp32`, `int8` |

---

## Auto-Discovery Logic

```
┌────────────────────────────────────────────────┐
│ Step 1: Load from model-engine-file (if set)   │
└──────────────────────┬─────────────────────────┘
                       ▼ not found
┌────────────────────────────────────────────────┐
│ Step 2: Auto-discover with standardized name   │
└──────────────────────┬─────────────────────────┘
                       ▼ not found
┌────────────────────────────────────────────────┐
│ Step 3: Build new engine, save for reuse       │
└────────────────────────────────────────────────┘
```

---

## Usage Examples

### Multiple Configurations (Auto-Cached)

```bash
/models/
  ├── model.onnx
  ├── model_b8_i640x640_sm120_rtx5090_trt10.7_fp16.engine
  ├── model_b4_i640x640_sm120_rtx5090_trt10.7_fp16.engine
  ├── model_b8_i1280x720_sm120_rtx5090_trt10.7_fp16.engine
  └── model_b8_i640x640_sm120_rtx5070_trt10.7_fp16.engine
```

Each configuration gets its own engine file. Switching between configurations loads instantly.

### Console Output

**First run:**
```
[INFO] Step 2: Looking for engine: model_b8_i640x640_sm120_rtx5090_trt10.7_fp16.engine
[INFO] No existing engine found
[INFO] Step 3: Building new engine from model files
[INFO] Engine built and saved successfully
```

**Subsequent runs:**
```
[INFO] Step 2: Looking for engine: model_b8_i640x640_sm120_rtx5090_trt10.7_fp16.engine
[INFO] Found existing engine file
[INFO] Successfully loaded engine (2.1 seconds)
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Auto-Discovery** | Finds existing engines without manual configuration |
| **Hardware-Specific** | Separate engines per GPU model (RTX 5090 vs RTX 5070) |
| **Version-Safe** | Includes TensorRT version to prevent incompatibility |
| **Multi-Config** | Different batch/input sizes coexist without overwriting |
| **Self-Healing** | Automatically rebuilds on any failure |
| **Backward Compatible** | `model-engine-file` param still works if specified |

---

## API Functions

```cpp
// Get GPU compute capability (e.g., "sm120")
std::string getGpuComputeCapability(int gpuId);

// Get GPU model name (e.g., "rtx5090")
std::string getGpuModelName(int gpuId);

// Get TensorRT version (e.g., "trt10.7")
std::string getTensorRTVersion();

// Generate standardized engine path
std::string generateEngineFilePath(
    const std::string& modelPath,
    int batchSize,
    int inputWidth,
    int inputHeight,
    int gpuId,
    int dlaCore,
    NvDsInferNetworkMode networkMode);

// Check if engine exists
bool engineFileExists(const std::string& enginePath);
```

---

## Files

| File | Description |
|------|-------------|
| `nvdsinfer_model_builder.h` | Function declarations |
| `nvdsinfer_model_builder.cpp` | Naming logic implementation |
| `nvdsinfer_context_impl.cpp` | Auto-discovery implementation |
