/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <initializer_list>
#include <iostream>
#include <vector>

constexpr unsigned int NB_COARSE_LABEL_AUDIO = 8;

constexpr auto LABELS_AUDIO = {
    "1_engine",
    "2_machinery-impact",
    "3_non-machinery-impact",
    "4_powered-saw",
    "5_alert-signal",
    "6_music",
    "7_human-voice",
    "8_dog",
    "1-1_small-sounding-engine",
    "1-2_medium-sounding-engine",
    "1-3_large-sounding-engine",
    "2-1_rock-drill",
    "2-2_jackhammer",
    "2-3_hoe-ram",
    "2-4_pile-driver",
    "3-1_non-machinery-impact",
    "4-1_chainsaw",
    "4-2_small-medium-rotating-saw",
    "4-3_large-rotating-saw",
    "5-1_car-horn",
    "5-2_car-alarm",
    "5-3_siren",
    "5-4_reverse-beeper",
    "6-1_stationary-music",
    "6-2_mobile-music",
    "6-3_ice-cream-truck",
    "7-1_person-or-small-group-talking",
    "7-2_person-or-small-group-shouting",
    "7-3_large-crowd",
    "7-4_amplified-speech",
    "8-1_dog-barking-whining"};

/* This is a sample audio neural network parsing function. */

extern "C"
{
    bool NvDsInferParseCustomAudio(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                   std::vector<NvDsInferAttribute> &attrList, std::string &attrString);
}

std::vector<unsigned int> index_giver_subcategory(const char *label);
