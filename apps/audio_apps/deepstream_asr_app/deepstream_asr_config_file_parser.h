/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __CONFIG_FILE_PARSER_H_
#define __CONFIG_FILE_PARSER_H_

#include <glib.h>
#include <stdio.h>

typedef struct __NvDsAudioConfig
{
  gboolean enable_playback;
  const char *asr_output_file_name;
  gboolean sync;
} NvDsAudioConfig;


typedef struct __NvDsAppConfig
{
  gboolean sync;
} NvDsAppConfig;


#endif
