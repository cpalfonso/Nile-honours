#!/bin/bash

## This script requires ffmpeg

scenario="hybrid"
framerate="5"  # 5 FPS (1s/Myr)

# Set the working directory to the directory containing this script
current_dir=$PWD
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
cd "${parent_path}" || exit


output_dir="${PWD}/../../results/${scenario}"
input_dir="${output_dir}/images"

input_elevations="${input_dir}/elevation_%03d.png"
input_erodep="${input_dir}/erodep_%03d.png"

output_elevations="${output_dir}/elevation.mp4"
output_erodep="${output_dir}/erodep.mp4"

create_animation () {
    local input="${1}"
    local output="${2}"
    local framerate="${3}"

    local temp_output
    temp_output="/tmp/$( hostname ).$$.mp4"

    ffmpeg -y \
        -framerate "${framerate}" \
        -pattern_type sequence \
        -i "${input}" \
        -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        -pix_fmt yuv420p \
        "${temp_output}"

    ffmpeg -y \
        -i "${temp_output}" \
        -vf fps=fps=30 \
        "${output}"
}

create_animation \
    "${input_elevations}" \
    "${output_elevations}" \
    "${framerate}"
create_animation \
    "${input_erodep}" \
    "${output_erodep}" \
    "${framerate}"


cd "${current_dir}" || exit
