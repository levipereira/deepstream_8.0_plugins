/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_asr_tts_app.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>


#define CONFIG_GROUP_SOURCE "source"
#define CONFIG_GROUP_ASR "asr"
#define CONFIG_GROUP_SINK "sink"

#define ERR_PRINT g_printerr

#undef DS_TRY
#undef DS_CATCH_ERROR
#undef DS_CATCH_ANY
#undef DS_CATCH_FULL

#define DS_TRY try

#define DS_CATCH_ERROR(type, fallback, fmt, ...)                            \
    catch (const type& e)                                                   \
    {                                                                       \
        ERR_PRINT(fmt ", catched " #type ":  %s", ##__VA_ARGS__, e.what()); \
        fallback;                                                           \
    }

#define DS_CATCH_ANY(fallback, fmt, ...)                         \
    catch (...)                                                  \
    {                                                            \
        ERR_PRINT(fmt ", catched unknown error", ##__VA_ARGS__); \
        fallback;                                                \
    }

#define DS_CATCH_FULL(fallback, fmt, ...)                            \
    DS_CATCH_ERROR(std::runtime_error, fallback, fmt, ##__VA_ARGS__) \
    DS_CATCH_ERROR(std::exception, fallback, fmt, ##__VA_ARGS__)     \
    DS_CATCH_ANY(fallback, fmt, ##__VA_ARGS__)

guint get_num_sources_yaml (gchar *config_file)
{
  guint num_users = 0;

  DS_TRY
  {
    YAML::Node node = YAML::LoadFile(config_file);
    while (1)
    {
      std::string source_name = CONFIG_GROUP_SOURCE + std::to_string(num_users);
      if (node[source_name])
      {
        num_users++;
      }
      else
      {
        break;
      }
    };
  }
  DS_CATCH_FULL(return 0, "Failed to parse YAML config file.");
  return num_users;
}

static gboolean parse_src_config_yaml (StreamCtx *sctx, YAML::Node node)
{
  DS_TRY
  {
    auto uriValue = node["uri"];

    if(uriValue)
    {
      gchar* filename = g_strdup(uriValue.as<std::string>().c_str());
      if (g_str_has_prefix(filename, "file:///") ||
          g_str_has_prefix(filename, "rtsp://")  )
      {
        sctx->uri = filename;
      }
      else
      {
        char *path = realpath(filename + 7, NULL);

        if (path == NULL)
        {
          g_printerr("Cannot find file with name[%s]\n", filename);
          return FALSE;
        }
        else
        {
          g_print("Input file [%s]\n", path);
          sctx->uri = g_strdup_printf("file://%s", path);
          free(path);
          free(filename);
        }
      }
    }
    else
    {
      g_printerr("Failed to parse source uri\n");
      return FALSE;
    }
  }
  DS_CATCH_FULL(return FALSE, "Failed to parse source config.");

  return TRUE;
}

static gboolean parse_asr_config_yaml (StreamCtx *sctx, YAML::Node node)
{
  DS_TRY
  {
    auto outFileValue = node["asr_output_file_name"];

    if(outFileValue) {
      sctx->audio_config.asr_output_file_name = g_strdup(outFileValue.as<std::string>().c_str());
    }
  }
  DS_CATCH_FULL(return FALSE, "Failed to parse asr config.");

  return TRUE;
}

static gboolean parse_sink_config_yaml (AppCtx *appctx, YAML::Node node)
{
  DS_TRY
  {
    auto playbackValue = node["enable_playback"];
    auto outFileValue = node["playback_output_file_name"];
    auto syncValue = node["sync"];

    if(playbackValue)
    {
      appctx->enable_playback = playbackValue.as<int>();
    }
    if(outFileValue)
    {
      appctx->playback_output_file_name = g_strdup(outFileValue.as<std::string>().c_str());
    }
    if(syncValue)
    {
      appctx->sync = syncValue.as<int>();
    }
  }
  DS_CATCH_FULL(return FALSE, "Failed to parse sink config.");

  return TRUE;
}

gboolean parse_config_file_yaml (AppCtx *appctx, gchar *config_file)
{
  gboolean ret = FALSE;
  guint num_users = 0;
  StreamCtx *sctx = NULL;

  DS_TRY
  {
    YAML::Node node = YAML::LoadFile(config_file);
    std::string sink_name = CONFIG_GROUP_SINK;

    for (num_users = 0; num_users < appctx->num_sources; num_users++)
    {
      std::string source_name = CONFIG_GROUP_SOURCE + std::to_string(num_users);
      std::string asr_name = CONFIG_GROUP_ASR + std::to_string(num_users);
      /* parse source group */
      if (node[source_name])
      {
        sctx = &appctx->sctx[num_users];
        ret = parse_src_config_yaml (sctx, node[source_name]);
        if (TRUE != ret)
        {
          return ret;
        }
      }
      else
      {
        g_printerr ("Failed to parse source%d config", num_users);
        return FALSE;
      }
      /* parse ASR group */
      if (node[asr_name])
      {
        sctx = &appctx->sctx[num_users];
        ret = parse_asr_config_yaml (sctx, node[asr_name]);
        if (TRUE != ret)
        {
          return ret;
        }
      }
      else
      {
        g_printerr ("Failed to parse asr%d config", num_users);
        return FALSE;
      }
    }

    /* parse common sink (renderer) configuration */
    if (node[sink_name])
    {
        ret = parse_sink_config_yaml (appctx, node[sink_name]);
        if (TRUE != ret)
        {
            return ret;
        }
    }
    else
    {
        g_printerr ("Failed to parse sink config");
        return FALSE;
    }
  }
  DS_CATCH_FULL(return FALSE, "Failed to parse YAML config file.");

  return TRUE;
}
