OUTPUT_PATH="experiments/results"
DATASET_PATH="dataset/scannet/scans"

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


overlapped_th=1e-3
max_correspondence_distance=0.03
knn_maxd=99999.0

trackable_opacity_th=0.09
overlapped_th2=1e-5
downsample_rate=5
keyframe_th=0.9
init_rate=1.0







### scene0000_00 ###
k_choice=1
knn_cov=15
keyframe_freq_tracking=30
depth_trunc=6.5

run_ "scene0000_00" "configs/scannet/scene0000_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0000_00.yaml"


### scene0059_00 ###
k_choice=3
knn_cov=15
keyframe_freq_tracking=40
depth_trunc=3.5

run_ "scene0059_00" "configs/scannet/scene0059_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0059_00.yaml"


### scene0106_00 ###
k_choice=1
knn_cov=15
keyframe_freq_tracking=45
depth_trunc=3.0

run_ "scene0106_00" "configs/scannet/scene0106_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0106_00.yaml"


### scene0169_00 ###
k_choice=1
knn_cov=16
keyframe_freq_tracking=20
depth_trunc=5.0

run_ "scene0169_00" "configs/scannet/scene0169_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0169_00.yaml"


### scene0181_00 ###
k_choice=5
knn_cov=15
keyframe_freq_tracking=25
depth_trunc=5.0

run_ "scene0181_00" "configs/scannet/scene0181_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0181_00.yaml"


#### scene0207_00 ###
k_choice=3
knn_cov=15
keyframe_freq_tracking=25
depth_trunc=6.0

run_ "scene0207_00" "configs/scannet/scene0207_00.txt" $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
      $trackable_opacity_th $overlapped_th2 $downsample_rate $k_choice $knn_cov $keyframe_freq_tracking $depth_trunc $init_rate  "configs_map/ScanNet/scene0207_00.yaml"



