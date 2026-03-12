OUTPUT_PATH="experiments/results"
DATASET_PATH="dataset/scannetpp/data"

str_pad() {

  local pad_length="$1" pad_string="$2" pad_type="$3"
  local pad length llength offset rlength

  pad="$(eval "printf '%0.${#pad_string}s' '${pad_string}'{1..$pad_length}")"
  pad="${pad:0:$pad_length}"

  if [[ "$pad_type" == "left" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "${pad:0:$length}$line"
    done

  elif [[ "$pad_type" == "both" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      llength="$(( length / 2 ))"
      offset="$(( llength + ${#line} ))"
      rlength="$(( llength + (length % 2) ))"
      echo -n "${pad:0:$llength}$line${pad:$offset:$rlength}"
    done

  else

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "$line${pad:${#line}:$length}"
    done

  fi
}

run_()
{
    local dataset=$1
    local config=$2
    local keyframe_th=$3
    local knn_maxd=$4
    local overlapped_th=$5
    local max_correspondence_distance=$6
    local trackable_opacity_th=$7
    local overlapped_th2=$8
    local downsample_rate=$9
    local k_choice=${10}
    local knn_cov=${11}
    local keyframe_freq_tracking=${12}
    local depth_trunc=${13}
    local init_rate=${14}
    local config_map_path=${15}
    
    echo "run $dataset"
    python -W ignore slam.py --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH\
                                    --keyframe_th $keyframe_th\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --save_results\
                                    --k_choice $k_choice\
                                    --knn_cov $knn_cov\
                                    --keyframe_freq_tracking $keyframe_freq_tracking\
                                    --depth_trunc $depth_trunc\
                                    --init_rate $init_rate\
                                    --config_map_path $config_map_path\

    wait
}



run_scannetpp()
{
    local keyframe_th=$1
    local knn_maxd=$2
    local overlapped_th=$3
    local max_correspondence_distance=$4
    local trackable_opacity_th=$5
    local overlapped_th2=$6
    local downsample_rate=$7
    local k_choice=$8
    local knn_cov=$9
    local keyframe_freq_tracking=${10}
    local depth_trunc=${11}
    local init_rate=${12}

    run_ "8b5caf3398" "configs/scannetpp/8b5caf3398.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate "configs_map/scannetpp/8b5caf3398.yaml"
    run_ "b20a261fdf" "configs/scannetpp/b20a261fdf.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate "configs_map/scannetpp/b20a261fdf.yaml"
    run_ "281bc17764" "configs/scannetpp/281bc17764.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate "configs_map/scannetpp/281bc17764.yaml"
    run_ "fb05e13ad1" "configs/scannetpp/fb05e13ad1.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate "configs_map/scannetpp/fb05e13ad1.yaml"
    run_ "2e74812d00" "configs/scannetpp/2e74812d00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate "configs_map/scannetpp/2e74812d00.yaml"
}



overlapped_th=1e-3
max_correspondence_distance=0.02
knn_maxd=99999.0

trackable_opacity_th=0.09
overlapped_th2=1e-5
downsample_rate=10
keyframe_th=0.9
init_rate=1.0
k_choice=5
knn_cov=10
keyframe_freq_tracking=10
depth_trunc=11

run_scannetpp $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
              $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate





