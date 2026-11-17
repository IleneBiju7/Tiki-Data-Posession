# Tiki-Data: ball possession per frame (More Than a Hack 2027, Phase 1)

Labels every frame of the BAR-ATH tactical video as H (Barça has the ball), A (Athletic has the ball), D (dead ball) or X (outside the graded halves).

Current output: `predictions/final.csv`. It passes the official format check (`possession_metrics.validate_submission`).

## How it works

1. **Ball.** The provided tracker has no ball on 41% of graded frames, so two ball detectors of our own run on every graded frame (`ball_detect_v2.py`). The first is a YOLO11s, started from Ultralytics' COCO weights and trained only on SoccerNet tracking: human ball boxes in other matches (`kaggle/ball_soccernet`). Our camera is far from the pitch, so the ball is much smaller than in SoccerNet's TV pictures, and the detector also learns from SoccerNet crops shrunk to a range of wide cameras, 0.25 to 0.75 of their size (`build_shrunk_mosaics.py --scales=0.25,0.75`). The second (`ball_st_k3.pt`, `kaggle/ball_st`) is the first trained 3 more epochs on the same pictures plus crops from SoccerTrack v2's panoramic videos, a wide camera like ours (see Training data). Each detector runs twice: on crops around the tracker's ball, then around its own first pass's ball.

   The merge (`fuse_simple.py`) takes, per frame, a trusted sighting of ours if there is one: a chain that moves like a ball, made of confident sightings or of weak ones lasting 1.48 s (first detector: conf >= 0.6, or >= 0.25 over that time; second: conf >= 0.7, or >= 0.45). Of several it takes the one nearest the previous ball. Otherwise it takes the provided tracker's ball, otherwise none. The thresholds come from SoccerNet test matches at TV scale and shrunk to 0.3, 0.4 and 0.6 (`tune_chains.py`, `ball_merge_eval.py --joint`), with a rule fixed before the search: find the most balls while the sightings used are right at least 95% of the time and there are at most 10 false sightings per 1,000 pictures without a ball, at every scale. The chosen setting finds 61.8% of the balls at 0.4 and 72.8% at TV scale (the first detector alone: 57.4% and 75.0%), is right at least 95.8% of the time (96.9%) and makes at most 8.6 false sightings per 1,000 (9.5). Our first attempt, keeping the first detector's own rule, found nothing that passed. So we searched both rules together, 432 settings, and the chosen one is the best of them on these same matches, which flatters it a little. On BAR-ATH the merged ball is there on 77.9% of graded frames, against 74.9% with the first detector alone.
2. **First stage.** Three LightGBM models (v43x-v45x) read team shape, speeds, our ball and the provided tracker's ball as separate inputs, stoppage clues from the players, pass chains and ball ownership, touches of the ball and whether they were controlled (`passes.py`), the ball's height from one camera (`ball_height.py`, through the provided homographies) and Smart Tagging. They give H/A/D probabilities for every frame (`features.py`, `train.py`). A hidden ball stays with the player who had it, up to 8 s with a goalkeeper (`ball_sticky.py`).
3. **Sequence networks.** The same features, 5 times a second, go through networks that see about 25 s either side of each moment: 7 dilated 1-D CNNs and 5 bidirectional GRUs, each from a different seed (`seq_export.py`, `seq_train.py`). A frame model sees one moment, these see how play unfolds around it, and they are the biggest single gain. The CNNs borrow two ideas from temporal action segmentation, the research field whose segmental F1 is what spellF1 measures: a second output that learns where possession changes hands (ASRF, Ishikawa et al., WACV 2021) and three refinement stages, each re-reading the previous stage's probabilities over about 50 s (MS-TCN, Farha and Gall, CVPR 2019).
4. **Player networks.** Four more networks that never read the ball, only the players' tracking, 5 times a second. Every player is a token (position, speed, team). Attention across the players gives a summary of each team, and a bidirectional GRU reads about 51 s of those summaries and gives H/A/D. A second output learns who has the ball: the owning team's player nearest the true ball of the open match, used only as a training target. Reading possession from player movement alone is the idea of PathCRF (Kim et al., KDD 2026, arXiv 2602.12080). We wrote our version from scratch (`kaggle/player_set`, `ps_predict.py`) and took no code or data from that paper. We left out its CRF, because our own decoding comes later. On their own these networks score well below the others, but adding them lifts the final (see Validation).
5. **Second stage.** A small LightGBM re-reads the averaged probabilities (trees 1/4, CNNs 1/3, GRUs 1/6, player networks 1/4) over the surrounding seconds plus context (ball visibility, ownership, touches, Smart Tagging) and corrects the timing (`stack.py`, `ctx_cache.py`).
6. **Decoding.** A Viterbi pass turns probabilities into clean spells (`decode_search.py`). Its switch costs are chosen on the held-out open matches.

## Training data

Every model is trained on external data only, as the organisers asked on 3 November 2026, and detector thresholds, stack weights and decoding costs are set on external data. No BAR-ATH label or frame is in any training set. Every model learns from other matches:

- DFL / IDSSE open data, 7 matches (Bassek et al. 2025, CC BY 4.0)
- Metrica Sports sample games 1-3
- PFF FC FIFA World Cup 2022, 64 matches (training only, every 10th frame; labels from PFF's ball-out, foul and restart events)
- SkillCorner open data (MIT licence), with PFF, to measure how a provider's tracker loses the ball: missing on 6-7% of live frames and 70-79% of dead frames, in gaps of median 4-6 s
- The ball detectors: the SoccerNet tracking train set (human-labelled balls), and for the second detector also SoccerTrack v2 (Hugging Face `atomscott/soccertrack-v2`, CC BY 4.0): 29,308 crops from 8 matches, its two benchmark test matches left out. No person drew those boxes. At a kick or dribble event in SoccerTrack's annotations the ball is at the acting player's feet, and our first detector boxes it there when its sightings agree over the neighbouring frames (`soccertrack_events.py`, `ball_pseudo_st.py`, `ball_pseudo_select.py`)

The first-stage LightGBM models, the network inputs, the networks and the ball detectors were trained on Kaggle (`kaggle/open_x_*`, `kaggle/seqx_*` and `kaggle/seqxb_*` (the same kernels on a second Kaggle account), `kaggle/player_set*`, `kaggle/ball_soccernet`, `kaggle/ball_st`, private notebooks) from open data and our code only. The second stage was fitted on our Mac, also from open data. All detection and prediction on BAR-ATH ran on our own Mac, and no BAR-ATH data went into training any model in this final.

The clean open tracking is degraded on purpose so it looks like a provider's tracking plus our detector (`train.py --external`). The provider-like tracker ball follows the SkillCorner and PFF numbers above. Our detector's noise spans its range on SoccerNet test (0.4-8.9 false sightings per 1,000 pictures without a ball). Airborne balls are seen through a generic wide camera, 15-60 m behind the touchline and 8-35 m high (`ball_height.generic_camera`), so they land where the camera ray meets the grass. Numbers nothing outside can measure (keeper duplicates, tracker slips, how often a flying ball is lost, Smart Tag timing) get a new draw from wide ranges for every copy. The ball-height estimator's two settings come from open matches seen through those generic cameras (`height_tune.py`: it finds 35% of flights above 2 m and calls 0.21% of balls on the ground airborne).

## Validation

Leave one match out on the 10 DFL and Metrica matches: train on the other 9 (plus PFF), score the held-out one with spellF1 from the organisers' `possession_metrics.py`, repeat. The "realistic" test copy is degraded as above, with a fixed seed.

| Model | Realistic spellF1 | What it is |
|---|---|---|
| Organisers' published baseline (their own test, for reference) | 0.2433 | The organisers' example solution (`baseline_smooth50`): the nearest player has the ball, short runs smoothed away. Their number on their test, not on our copies. |
| 4 player networks averaged | 0.5768 | Ball-free networks (step 4), seeds 7, 100, 200 and 300, each 0.566-0.574 alone. |
| v45x alone | 0.5775 | v43x without the ownership inputs. |
| v44x alone | 0.5788 | v43x's inputs, learnt from one copy with a detector-like ball and one with the clean ball. |
| v43x alone (best LightGBM model) | 0.5802 | One LightGBM frame model with 150 inputs (step 2 above), trained on the 9 other open matches plus 64 PFF matches, degraded as above. |
| 5 GRUs averaged | 0.6575 | Bidirectional GRUs, seeds 20-24. |
| 7 CNNs averaged | 0.6839 (+0.104 on v43x) | Dilated 1-D CNNs with the boundary output and three refinement stages, seeds 7-9 and 11-14. |
| v43x-v45x + 12 networks + second stage | 0.6850 | The same stack without the player networks. Per held-out match 0.60-0.77. |
| Final: v43x-v45x + 12 networks + 4 player networks + second stage | **0.6939** | Per held-out match 0.61-0.77. |

We set the bar before running the player networks: a model stays only if this score rises by more than 0.003. In the average they get weight 3, against 1 for each tree, 4 for the CNNs and 2 for the GRUs. Weights 1, 2, 3 and 4 give 0.6889, 0.6905, 0.6939 and 0.6934. The weight was picked on these same held-out matches, which flatters the best one a little, so the real gain is probably a little under +0.009.

How the score falls as the ball goes missing (`fold_train.py`, then `degradation_test.py`): v43x's 10 fold models (retrained on the Mac) and the organisers' baseline (the method of their `starter/baseline.py`: the team of the player nearest the tracker's ball, D where the tracker has no ball, then their majority vote over ±50 frames as in `baseline_smooth50`), on each held-out open match's clean tracking, then with the ball removed in gaps (median 6 s, as in SkillCorner's tracking) until a share of the played frames has no ball, then on the realistic copy:

| Test copy | v43x alone | Organisers' baseline |
|---|---|---|
| Clean tracking | 0.6325 | 0.3366 |
| Ball missing on 10% of played frames | 0.6093 | 0.3037 |
| 20% | 0.5873 | 0.2787 |
| 30% | 0.5588 | 0.2545 |
| 40% | 0.5305 | 0.2307 |
| Realistic copy | 0.5795 | 0.3161 |

Both lose about 0.026 for every 10% of frames without a ball, but from very different levels: with 40% of the ball gone, v43x keeps 84% of its clean score and the baseline 69%. Our merged ball is missing on 22.1% of BAR-ATH's graded frames.

Tried and dropped, on the same copies. All of these came before the player networks, so "the final" in this list is the 0.6850 stack:

- Team-swap averaging (`predict_stack.py --swap-tta=seq`, `tta_cv.py`, `tta_eval.py`; networks retrained by `kaggle/abx_seqw_m7`, `kaggle/seqw_*` and `kaggle/seqwb_*` on `kaggle/abx_export-sw`): every network also reads the match with the two teams' identities swapped, H and A are swapped back, and the two readings are averaged. It is a real gain on the same networks: one CNN alone gains 0.010-0.020 (seed 7: 0.6615 to 0.6713), the 7 CNNs 0.6782 to 0.6841 and the whole stack 0.6816 to 0.6854. But it needs every network retrained so its held-out matches are read both ways, and the retrained networks without it scored 0.6816, against 0.6850 for the same seeds in the final: run-to-run noise is about as large as the gain. The final with it would score 0.6854, +0.0004, and with the earlier 12 networks added 0.6849. The trees gain nothing (v43x 0.5842 without, 0.5831 with): they already learn from both team orders.
- SkillCorner open data (20 matches, broadcast tracking, MIT licence) as extra training matches: one CNN 0.6503 against 0.6615 without (with team-swap averaging 0.6664 against 0.6713), and v43x 0.5593 against 0.5802 (`kaggle/abx_v43xc`).
- A wider grid of decoding costs: +0.0000.
- A decoder that knows how long spells last (`duration_decode.py`, an explicit-duration Viterbi as in hidden semi-Markov models, with spell-length hazards from the 9 training matches): 0.6845.
- The ball detector trained for 12 epochs instead of 6: no threshold setting met the SoccerNet rule above.
- Ball actions with the team that made them as second-stage inputs (`actions.py`, the idea behind T-DEED's team ball-action spotting on broadcast video), tested before building any video model. With every on-ball action of the DFL event data, exact, the 7 DFL matches gain +0.019 (0.6572 to 0.6763). With errors like a video model's today (60% of actions found, 1 false per minute, 90% given to the right team) they lose 0.0007, and they gain only from about 80-90% found. We did not build it.

The open matches have no video, so the detectors never run in this test, and adding the second detector does not change the scores above. We measure our ball's accuracy on SoccerNet (step 1).

## Run it

Python 3.11, then `pip install -r requirements.txt`. The organisers' repository must sit next to this one as `official-repo/`, because the code imports its `evaluation/possession_metrics.py`. Built and checked on an Apple M2 (macOS, PyTorch on MPS): re-running the prediction gives the same `final.csv`, byte for byte, and re-running both detectors on the first 400 graded frames gives the same sightings. The prediction also gives the same `final.csv`, byte for byte, with PyTorch on the CPU. The scripts in this folder pick their device themselves (an Nvidia GPU, else Apple's MPS, else the CPU); those in `kaggle/` are kept as they ran on Kaggle GPUs.

The competition files are not in this repo. Put them in a folder with a `tactical/` subfolder and point to it:

```
export MTAH_DATA_DIR=/path/to/mtah2027
```

Besides `predictions/final.csv`, the only thing in these files made from the competition files is `detections/`: our two detectors' frame-level ball sightings, which the organisers said on 3 November that teams may provide. With them the video steps can be skipped. Cached features are not in the repo, so the first run builds them.

**Predict with the trained models, from the shipped sightings.** We ran these commands on a fresh clone, with a data folder holding only `tactical/`: `predictions/final.csv` came back byte for byte in about 10 minutes on an Apple M2.

```
python atd_cache.py
mkdir -p "$MTAH_DATA_DIR/cache" && cp detections/ball_detections_*.csv.gz "$MTAH_DATA_DIR/cache/"
python fuse_simple.py k3 m3 --anchor-conf=0.6 --chain-conf=0.25 --chain-seconds=1.48 --also=t3,0.7,0.45,1.48
python ball_height.py fused_m3
CNN=seq_kaggle_xm7+seq_kaggle_xm8+seq_kaggle_xm9+seq_kaggle_xm11+seq_kaggle_xm12+seq_kaggle_xm13+seq_kaggle_xm14
GRU=seq_kaggle_xg20+seq_kaggle_xg21+seq_kaggle_xg22+seq_kaggle_xg23+seq_kaggle_xg24
PS=seq_kaggle_ps_f7+seq_kaggle_ps_s100+seq_kaggle_ps_s200+seq_kaggle_ps_s300
python predict_stack.py final --stage1=v43x+v44x+v45x --seq=$CNN,$GRU --seq-weight=4,2 --ps=$PS --ps-weight=3 --stack=v43x+v44x+v45x+seqMx@4+seqGx@2+seqPS4@3_c_p_s_tok_z --ctx --passes --trk --own --contacts --touch --ball=fused_m3 --decode=auto
```

**To run the detectors on the video as well,** replace the `cp` line with these. They take about 11 hours on an Apple M2, because both detectors run twice on every graded frame:

```
python ball_detect_v2.py 1 ball_soccernet_k.pt k1 tracker
python fuse_simple.py k1 k1 --anchor-conf=0.6 --chain-conf=0.25 --chain-seconds=1.0
python ball_detect_v2.py 1 ball_soccernet_k.pt k3 k1
python ball_detect_v2.py 1 ball_st_k3.pt t1 tracker
python fuse_simple.py t1 t1 --anchor-conf=0.6 --chain-conf=0.45 --chain-seconds=1.48
python ball_detect_v2.py 1 ball_st_k3.pt t3 t1
```

**Retrain everything:** `./run_final.sh`. It takes about a day on one Mac, plus about 16 GPU hours for the networks; we ran the networks and first-stage trees on Kaggle, with this code before unused options were removed, and the steps separately rather than as one script. It also needs the open datasets: DFL (downloaded by kloppy) and Metrica's sample-data repository in `../data/metrica-sample-data`, both converted by `convert_open.py` (Metrica's labels come from its event data, `metrica_labels.py`), PFF World Cup 2022 in `../data/open/pff`, SoccerNet tracking in `../data/open/soccernet` and SoccerTrack v2 in `../data/open/soccertrack`.

## Credits

- DFL / IDSSE open data: Bassek, Rein, Weber and Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12, 195 (2025), CC BY 4.0.
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data
- PFF FC FIFA World Cup 2022 dataset: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data
- SoccerNet tracking: Cioppa et al., "SoccerNet-Tracking", CVPR Workshops 2022 (https://github.com/SoccerNet/sn-tracking). Training on Kaggle used the copy `theoviel/soccernet-tracking`.
- SoccerTrack v2: https://huggingface.co/datasets/atomscott/soccertrack-v2, CC BY 4.0 (arXiv 2508.01802).
- SkillCorner open data: https://github.com/SkillCorner/opendata
- Ultralytics YOLO11, with COCO weights as the detectors' starting point: AGPL-3.0.
- Claude (Anthropic), as a coding assistant: it helped write, run and review the code. No data was uploaded to it. Claude does not run as part of the pipeline that makes the predictions.

## Known weak spots

- When the provided tracker's ball is on the wrong player and our detector sees the ball only now and then, the wrong ball stays.
- Track slots that follow an opponent for a while are not corrected, and players the tracker misses near the ball are not added. A same-team model trained on other matches could do both.
- Fouls and offsides are read from player behaviour only.
- The broadcast recording comes with its own tracking. `broadcast_sync.py` lines it up with the tactical recording from the two trackings alone, and its ball could fill gaps where ours has none. It is not in the final: the open matches have no second recording, so it cannot be tested.
