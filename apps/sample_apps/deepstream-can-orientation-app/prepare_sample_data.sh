#!/bin/bash
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

set -e

sample_tar=/opt/nvidia/deepstream/deepstream/samples/streams/sample_cans_jpg.tbz2

echo "uncompress  ${sample_tar}"
tar -pxvf ${sample_tar} -C .

from_dir=data/sample0
[ "$(ls -A ${from_dir})" ] || \
    { echo "${from_dir} is empty, please download test images into the folder.";  exit -1; }

ref_image=data/reference_sample.jpg
[ -f "${ref_image}" ] || \
    { echo "${ref_image} is not exist, please download reference image.";  exit -1; }

to_dir=data/sample1
echo "Copy and format original test image files into folder ${to_dir}"
mkdir -p ${to_dir}
sidx=0
for file_name in `ls $from_dir`;
do
    tofile=$(printf "${to_dir}/test_sample_%04d.jpg" $sidx)
    cp -f ${from_dir}/${file_name} ${tofile}
    ((sidx=sidx+1))
done

png_dir=data/sample2
echo "Convert JPG files into PNG into folder ${png_dir}"
mkdir -p ${png_dir}
sidx=0
for file_name in `ls $from_dir`;
do
    png_file=$(printf "${png_dir}/test_sample_%04d.png" $sidx)
    gst-launch-1.0 -e filesrc location="${from_dir}/${file_name}" ! \
        jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! pngenc ! filesink location="${png_file}"
    ((sidx=sidx+1))
done

grey_sample=data/test_samples_raw.grey
echo "Generate raw GREY test sample file ${grey_sample}"
gst-launch-1.0 -e multifilesrc location="${to_dir}/test_sample_%04d.jpg" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! \
    filesink location=${grey_sample}

rgba_sample=data/test_samples_raw.rgba
echo "Generate raw RGBA test sample file ${rgba_sample}"
gst-launch-1.0 -e multifilesrc location="${to_dir}/test_sample_%04d.jpg" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=RGBA" ! \
    filesink location=${rgba_sample}

grey_ref_sample=data/reference_sample.grey
echo "Generate GREY reference sample file ${grey_ref_sample}"
gst-launch-1.0 -e filesrc location="${ref_image}" ! \
    jpegdec ! videoconvert ! "video/x-raw, format=GRAY8" ! \
    filesink location=${grey_ref_sample}

echo "Data is ready"
