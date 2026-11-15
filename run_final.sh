#!/bin/bash
# Rebuild the final submission from scratch, with every model trained on external data only
# (organisers, 3 Nov 2026): our ball (two detectors, SoccerNet and SoccerTrack v2), three first-stage LightGBM models, an ensemble of
# sequence networks and four ball-free player networks trained on open matches, a second stage with ball-ownership and
# touch context, decoding.
# Data stays local: set MTAH_DATA_DIR to the folder holding tactical/ (see atd_cache.py).
# Open training matches go to ../data/open/converted and ../data/open/converted_pff (convert_open.py).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}

# 1) inputs
$PY atd_cache.py                       # provided tracking -> numpy cache
$PY convert_open.py dfl                # open training matches (other games only)
$PY convert_open.py metrica
$PY convert_open.py pff                # PFF FC World Cup 2022 (64 matches) -> ../data/open/converted_pff

# 2) the first ball detector, SoccerNet only: crops of the SoccerNet train clips (human ball boxes, other matches), the same
#    crops shrunk to a range of wide cameras (0.25-0.75 of their size, not our camera's own scale), YOLO11s for 6
#    epochs. We trained it on a Kaggle GPU with the same recipe (kaggle/ball_soccernet); models/weights/ball_soccernet_k.pt
#    is shipped. Then its checks on SoccerNet test (human labels) at TV scale and shrunk to 0.3, 0.4 and 0.6, and the
#    merge thresholds from those checks only (tune_chains.py: confident 0.6, weak 0.25 over 1.0 s).
$PY build_soccernet_crops.py                                     # -> ../data/open/soccernet/crops
$PY build_shrunk_mosaics.py 3000 --scales=0.25,0.75              # -> ../data/open/soccernet/crops_multiscale
$PY train_ball_detector.py ../data/open/soccernet/ball_soccernet.yaml ball_soccernet_k 6 yolo11s.pt 4
mkdir -p cache                                                   # the detector checks save their sightings here
$PY tv_eval.py ball_soccernet_k
$PY shrunk_soccernet.py --weights=ball_soccernet_k
for s in 0.3 0.6; do $PY shrunk_soccernet.py --weights=ball_soccernet_k --shrink=$s; done
$PY tune_chains.py ball_soccernet_k

#    The second detector: the first one trained 3 more epochs on the same crops plus SoccerTrack v2 crops (CC BY 4.0,
#    ../data/open/soccertrack; the ball at the acting player's feet at kick and dribble events, boxed by the first detector
#    where its sightings agree, no person decides a box; the two benchmark test matches left out). We made the crops with
#    st_pseudo_all.sh (soccertrack_events.py, ball_pseudo_st.py, ball_pseudo_select.py) and trained on a Kaggle GPU
#    (kaggle/ball_st); models/weights/ball_st_k3.pt is shipped. Then its checks on SoccerNet test, its own first-pass
#    thresholds, and the merge thresholds of both detectors together, from those checks only.
$PY tv_eval.py ball_st_k3
$PY shrunk_soccernet.py --weights=ball_st_k3
for s in 0.3 0.6; do $PY shrunk_soccernet.py --weights=ball_st_k3 --shrink=$s; done
$PY tune_chains.py ball_st_k3 --wide
$PY ball_merge_eval.py ball_soccernet_k ball_st_k3 --joint

# 3) our ball on BAR-ATH: each detector runs two passes, so nothing depends on an earlier run (crops centred on the
#    tracker's ball, then on that detector's first pass's ball), the plain merge of both second passes with the thresholds
#    above, then the ball's height from one camera (the provided homographies; WIN and FLIGHT_GAIN in ball_height.py come
#    from height_tune.py on open matches seen through generic cameras). The second passes' sightings are shipped in
#    detections/ (copy them to $MTAH_DATA_DIR/cache to skip the detector runs).
$PY height_tune.py
$PY ball_detect_v2.py 1 ball_soccernet_k.pt k1 tracker      # -> cache/ball_detections_k1.csv.gz
$PY fuse_simple.py k1 k1 --anchor-conf=0.6 --chain-conf=0.25 --chain-seconds=1.0    # -> cache/BAR-ATH_ball_fused_k1.npy
$PY ball_detect_v2.py 1 ball_soccernet_k.pt k3 k1           # -> cache/ball_detections_k3.csv.gz
$PY ball_detect_v2.py 1 ball_st_k3.pt t1 tracker            # -> cache/ball_detections_t1.csv.gz
$PY fuse_simple.py t1 t1 --anchor-conf=0.6 --chain-conf=0.45 --chain-seconds=1.48   # -> cache/BAR-ATH_ball_fused_t1.npy
$PY ball_detect_v2.py 1 ball_st_k3.pt t3 t1                 # -> cache/ball_detections_t3.csv.gz
$PY fuse_simple.py k3 m3 --anchor-conf=0.6 --chain-conf=0.25 --chain-seconds=1.48 --also=t3,0.7,0.45,1.48
                                                            # -> cache/BAR-ATH_ball_fused_m3.npy (the ball we predict with)
$PY ball_height.py fused_m3                                 # -> cache/BAR-ATH_ball_height_fused_m3.npy

# 4) first stage: three LightGBM frame models, each with leave-one-match-out CV on the 10 open matches. --external:
#    every degradation number from other providers' tracking (SkillCorner, PFF), SoccerNet or wide random ranges.
#    We ran these on Kaggle CPUs (kaggle/open_x_v43x, _v44x, _v45x).
F="--mixed-tags --sticky --hard-test --big --smooth --pff --trk --keeper --sp-calib --contacts --height --external"
$PY train.py $F --det-noise --tag=v43x
$PY train.py $F --mixed-noise --tag=v44x
$PY train.py $F --det-noise --no-own --tag=v45x

# 5) sequence networks over the same features, 5 steps per second: dilated 1-D CNNs (7 seeds) and bidirectional GRUs
#    (5 seeds), same leave-one-match-out split. The CNNs also learn where the label changes and have 3 refinement
#    stages (--boundary=1 --stages=3, from action segmentation: ASRF and MS-TCN). The export ran on a Kaggle CPU
#    (kaggle/open_x_export-x), the networks on Kaggle GPUs (kaggle/seqx_*, public matches only, the same seq_train.py);
#    they are shipped in models/seq_kaggle_x*/ (cv_seq.npz = held-out probabilities, seq_final.pt = trained on all
#    matches). To retrain one here:
#    $PY seq_train.py --data=models/seq_x --out=models/seq_kaggle_xm7 --seed=7 --boundary=1 --stages=3
#    (GRUs: --arch=gru without --boundary/--stages; GPU or Apple MPS)
$PY seq_export.py --mixed-tags --sticky --smooth --pff --trk --keeper --sp-calib --det-noise --contacts --height --external
CNN=seq_kaggle_xm7+seq_kaggle_xm8+seq_kaggle_xm9+seq_kaggle_xm11+seq_kaggle_xm12+seq_kaggle_xm13+seq_kaggle_xm14   # seeds 7-9, 11-14
GRU=seq_kaggle_xg20+seq_kaggle_xg21+seq_kaggle_xg22+seq_kaggle_xg23+seq_kaggle_xg24                                # seeds 20-24
cvs() { for d in ${1//+/ }; do echo models/$d/cv_seq.npz; done; }
$PY seq_cv.py seqMx v43x $(cvs $CNN)   # -> models/cv_probs_seqMx_hard.npz, the CNN seeds averaged
$PY seq_cv.py seqGx v43x $(cvs $GRU)   # -> models/cv_probs_seqGx_hard.npz, the GRU seeds averaged

# 6) player networks: every player a token, the ball never read (the idea of PathCRF, Kim et al., KDD 2026; written from
#    scratch, its CRF left out), same leave-one-match-out split and degraded copies. 4 seeds on Kaggle GPUs (kaggle/player_set,
#    _f7, _s100, _s200, _s300; public matches only); they are shipped in models/seq_kaggle_ps_*/ (cv_seq.npz, ps_final.pt).
#    To retrain one here (seed and what to run are set at the top of the file; set PS_OUT, GPU recommended):
#    mkdir -p models/seq_kaggle_ps_f7 && PS_OUT=models/seq_kaggle_ps_f7 $PY kaggle/player_set/player_set.py
PS=seq_kaggle_ps_f7+seq_kaggle_ps_s100+seq_kaggle_ps_s200+seq_kaggle_ps_s300   # seeds 7, 100, 200, 300
$PY seq_cv.py seqPS4 v43x $(cvs $PS)   # -> models/cv_probs_seqPS4_hard.npz, the player networks averaged

# 7) second stage: stack on the held-out probabilities (CNNs 4/12, GRUs 2/12, player networks 3/12 of the average, each
#    tree 1/12) + context, then fit on all matches
$PY ctx_cache.py --sticky --mixed-tags --passes --smooth --trk --keeper --sp-calib --own --contacts --height --touch --external
$PY stack.py v43x+v44x+v45x+seqMx@4+seqGx@2+seqPS4@3 --ctx --passes --smooth --trk --own --height --contacts --touch --external --save

# 8) predict BAR-ATH
$PY predict_stack.py final --stage1=v43x+v44x+v45x --seq=$CNN,$GRU --seq-weight=4,2 --ps=$PS --ps-weight=3 \
    --stack=v43x+v44x+v45x+seqMx@4+seqGx@2+seqPS4@3_c_p_s_tok_z --ctx --passes --trk --own --contacts --touch --ball=fused_m3 \
    --decode=auto
