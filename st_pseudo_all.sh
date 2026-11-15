#!/bin/zsh
# Pseudo ball boxes on SoccerTrack v2 (open data, CC BY 4.0), half by half: fetch the half's panoramic video from
# Hugging Face (the next one downloads while the current one is read), run ball_pseudo_st.py, then delete the video
# again (a full copy of the dataset is kept elsewhere). One half of each training match first, then the two benchmark
# test matches (M7 128057, M9 132831: never trained on), then the remaining halves.
# Log: ../data/open/soccertrack/pseudo/run.log
cd "$(dirname "$0")"
P=${PY:-python}
ROOT="$(cd .. && pwd)/data/open/soccertrack"
ORDER=(117093:1st 117092:1st 118575:1st 118576:1st 118577:1st 118578:1st 128058:1st 132877:1st 128057:1st 132831:1st
       117093:2nd 117092:2nd 118575:2nd 118576:2nd 118577:2nd 118578:2nd 128058:2nd 132877:2nd 128057:2nd 132831:2nd)
video() { echo "$ROOT/videos/$1/$1_panorama_$2_half.mp4" }
fetch() {
  [[ -f "$(video $1 $2)" ]] && return 0
  $P -I -c "from huggingface_hub import hf_hub_download as d; d('atomscott/soccertrack-v2', 'videos/$1/$1_panorama_$2_half.mp4', repo_type='dataset', local_dir='$ROOT')" > /dev/null 2>&1
}
mkdir -p "$ROOT/pseudo"
next=""
for i in {1..${#ORDER}}; do
  m=${ORDER[i]%%:*}; h=${ORDER[i]##*:}
  if [[ -n $next ]]; then wait $next; else fetch $m $h; fi
  if (( i < ${#ORDER} )); then fetch ${ORDER[i+1]%%:*} ${ORDER[i+1]##*:} & next=$!; else next=""; fi
  if [[ -f "$ROOT/pseudo/${m}_${h}/dets.csv" ]]; then echo "$(date +%H:%M) $m $h already done"; rm -f "$(video $m $h)"; continue; fi
  if [[ ! -f "$(video $m $h)" ]]; then echo "$(date +%H:%M) $m $h: video missing, skipped"; continue; fi
  echo "$(date +%H:%M) $m $h: $($P ball_pseudo_st.py $m $h 2>&1 | grep -v -i warning | tail -1)"
  rm -f "$(video $m $h)"
done
echo "$(date +%H:%M) all halves done"
