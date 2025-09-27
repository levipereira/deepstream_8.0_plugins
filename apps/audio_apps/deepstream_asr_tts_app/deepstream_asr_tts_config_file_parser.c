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

#include "deepstream_asr_tts_app.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CONFIG_GROUP_SOURCE "source"
#define CONFIG_GROUP_ASR "asr"
#define CONFIG_GROUP_SINK "sink"

#define CHECK_PARSE_ERROR(error) \
    if (error) { \
        g_printerr ("%s", error->message); \
        goto done; \
    }

static guint get_num_sources_cfg (gchar *config_file)
{
  GError *error = NULL;
  gchar **groups = NULL;
  gchar **group;
  guint num_users = 0;

  GKeyFile *cf = g_key_file_new ();

  if (!g_key_file_load_from_file (cf, config_file, G_KEY_FILE_NONE, &error))
  {
    if (error) {
      g_printerr ("Failed to load config file: %s, %s",config_file, error->message);
      g_error_free (error);
    } else {
      g_printerr ("Failed to load config file: %s", config_file);
    }
    return 0;
  }

  groups = g_key_file_get_groups (cf, NULL);

  for (group = groups; *group; group++) {

   if (!strncmp (*group, CONFIG_GROUP_SOURCE, sizeof (CONFIG_GROUP_SOURCE) - 1)) {
      num_users++;
    }
  }

  if (cf) {
    g_key_file_free (cf);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }

  return num_users;
}

guint get_num_sources (gchar *config_file)
{
  if (!config_file)
  {
    g_printerr ("Config file name not available\n");
    return 0;
  }

  if (g_str_has_suffix (config_file, ".yml") ||
          g_str_has_suffix (config_file, ".yaml"))
  {
    return get_num_sources_yaml (config_file);
  }
  else
  {
    return get_num_sources_cfg (config_file);
  }
}

static gboolean parse_src_config (StreamCtx *sctx, GKeyFile *key_file, gchar *config_file, gchar *group)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);

  for (key = keys; *key; key++)
  {
    if (!g_strcmp0 (*key, "uri"))
    {
      gchar *filename = (gchar *)g_key_file_get_string(key_file, group, "uri", &error);
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
          printf("cannot find file with name[%s]\n", filename);
          if (filename) {
            g_free(filename);
          }
          goto done;
        }
        else
        {
          printf("Input file [%s]\n", path);
          sctx->uri = g_strdup_printf("file://%s", path);
          free(path);
        }
      }
      CHECK_PARSE_ERROR(error);
      if (filename) {
          g_free(filename);
      }
    }
  }
  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

static gboolean parse_asr_config(StreamCtx *sctx, GKeyFile *key_file, gchar *config_file, gchar *group)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);

  for (key = keys; *key; key++)
  {
    if (!g_strcmp0 (*key, "asr_output_file_name"))
    {
      sctx->audio_config.asr_output_file_name  = (gchar *)g_key_file_get_string (key_file, group, "asr_output_file_name", &error);
      CHECK_PARSE_ERROR (error);
    }
  }

  if (error) {
    g_error_free (error);
  }

  if (keys) {
    g_strfreev (keys);
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

static gboolean parse_sink_config (AppCtx *apptx, GKeyFile *key_file, gchar *config_file, gchar *group)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (key_file, group, NULL, &error);


  for (key = keys; *key; key++)
  {
    if (!g_strcmp0 (*key, "enable_playback"))
    {
      apptx->enable_playback  = g_key_file_get_integer (key_file, group, "enable_playback", &error);
      CHECK_PARSE_ERROR (error);
    }

    if (!g_strcmp0 (*key, "playback_output_file_name"))
    {
      apptx->playback_output_file_name  = (gchar *)g_key_file_get_string (key_file, group, "playback_output_file_name", &error);
      CHECK_PARSE_ERROR (error);
    }

    if (!g_strcmp0 (*key, "sync"))
    {
      apptx->sync  = g_key_file_get_integer (key_file, group, "sync", &error);
      CHECK_PARSE_ERROR (error);
    }
  }
  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

static gboolean parse_config_file_cfg (AppCtx *appctx, gchar *config_file)
{
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;
  int i = 0;
  StreamCtx *sctx = NULL;

  GKeyFile *cf = g_key_file_new ();

  if (!g_key_file_load_from_file (cf, config_file, G_KEY_FILE_NONE, &error))
  {
    g_printerr ("Failed to load config file: %s, %s",config_file, error->message);
    goto done;
  }

  groups = g_key_file_get_groups (cf, NULL);

  for (group = groups; *group; group++) {

    /* parse source group */
    if (!strncmp (*group, CONFIG_GROUP_SOURCE, sizeof (CONFIG_GROUP_SOURCE) - 1))
    {
      sctx = &appctx->sctx[i];
      ret = parse_src_config (sctx, cf, config_file, *group);
      if (TRUE != ret)
      {
        goto done;
      }
    }

    /* parse ASR group */
    /* Increment the stream counter when both source and ASR groups are present */
    if (!strncmp (*group, CONFIG_GROUP_ASR, sizeof (CONFIG_GROUP_ASR) - 1))
    {
      sctx = &appctx->sctx[i];
      ret = parse_asr_config (sctx, cf, config_file, *group);
      if (TRUE != ret)
      {
        goto done;
      }
      i++;
    }

    /* parse sink (renderer) group */
    if (!strncmp (*group, CONFIG_GROUP_SINK, sizeof (CONFIG_GROUP_SINK) - 1))
    {
      ret = parse_sink_config (appctx, cf, config_file, *group);
      if (TRUE != ret)
      {
        goto done;
      }
    }

  }
done:

  if (cf) {
    g_key_file_free (cf);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }

  return ret;
}

gboolean parse_config_file (AppCtx *appctx, gchar *config_file)
{
  if (!config_file)
  {
    g_printerr ("Config file name not available\n");
    return FALSE;
  }

  if (g_str_has_suffix (config_file, ".yml") ||
          g_str_has_suffix (config_file, ".yaml"))
  {
    return parse_config_file_yaml (appctx, config_file);
  }
  else
  {
    return parse_config_file_cfg (appctx, config_file);
  }
}
