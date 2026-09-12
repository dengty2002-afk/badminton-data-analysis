"""Run a local experimental HitNet checkpoint on cached features; no Gold required."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch
from hitnet import HitNet, build_features, decode, bst_windows
from run_experiment import HERE, jsonl, dump, sha, predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():
        p.error('output must be a new directory inside this experiment')
    rows = jsonl(args.frames)
    if len({r['video_id'] for r in rows}) != 1:
        p.error('one source video per invocation is required')
    cal = json.loads(args.calibration.read_text(encoding='utf-8'))
    if cal['video_id'] != rows[0]['video_id']:
        p.error('calibration and frames video IDs do not match')
    fps = round(1/np.median(np.diff([r['timestamp_sec'] for r in rows])),6)
    if not 29.8 <= fps <= 30.2:
        p.error('local checkpoint supports approximately 30 FPS only; resampling is not validated')
    x, ids, raw = build_features(rows,cal)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(args.checkpoint,map_location=device,weights_only=True)
    if state.get('origin') != 'locally_trained_not_official':
        p.error('expected isolated local reproduction checkpoint')
    model = HitNet().to(device)
    model.load_state_dict(state['state_dict'])
    started = time.perf_counter()
    probs = predict(model,x,device)
    hits,sides,meta = decode(probs,fps,raw,cal)
    elapsed = time.perf_counter()-started
    events = [{'video_id':rows[0]['video_id'],'event_id':f'hitnet_local_{i:05d}',
               'candidate_frame':int(ids[0])+h,'predicted_hitter':'lower' if s==1 else 'upper',
               'rally_id':'unknown','event_source':'monotrack_architecture_local_training'}
              for i,(h,s) in enumerate(zip(hits,sides))]
    out.mkdir(parents=True)
    jsonl(out/'events.jsonl',events)
    jsonl(out/'bst-windows.jsonl',bst_windows(events,fps,int(ids[0]),int(ids[-1])))
    np.save(out/'probabilities.npy',probs)
    summary = {'model_origin':state['origin'],'checkpoint_sha256':sha(args.checkpoint),
               'frames_sha256':sha(args.frames),'calibration_sha256':sha(args.calibration),
               'video_id':rows[0]['video_id'],'fps':fps,'frame_count':len(rows),
               'hit_count':len(events),'device':device,'inference_and_decode_seconds':elapsed,
               'decoder':meta,'uses_gold':False}
    dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
