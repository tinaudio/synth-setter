#!/usr/bin/env bash
## declare an array variable
declare -a datasets=("full" "simple" "nsynth" "fsd")

mkdir aggregated_audio
## now loop through the above array
for dataset in "${datasets[@]}"
do
  mkdir -p "aggregated_audio/$dataset"
  audio_dir="scripts/audio_dirs/${dataset}.txt"
  while read dir; do
    model=$( basename $dir )
    model=$( echo $model | cut -d'-' -f1 )

    mkdir -p "aggregated_audio/$dataset/$model"

    while read sample; do
      cp -R "$dir/audio/$sample" "aggregated_audio/$dataset/$model"
    done < "scripts/sample_lists/${dataset}.txt"
  done < "$audio_dir"
done
