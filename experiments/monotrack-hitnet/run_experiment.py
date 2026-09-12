"""Run the frozen isolated HitNet experiment; outputs never enter production."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from badminton_pipeline.evaluation.hit_events import read_gold, evaluate_candidates
from hitnet import HitNet, build_features, window_labels, decode, bst_windows


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def jsonl(path, rows=None):
    if rows is None:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def cases():
    specs = [
        ('train', 'vid_2e2258f261aa', 'full-local-v0.3-play-gate-dev'),
        ('train', 'vid_a598228fbc11', 'holdout-a598-play-gate-dev'),
        ('validation', 'vid_576b0c1ab370', 'validation-576b-play-gate-dev'),
    ]
    specs += [('test', p.name, f'play-state-prospective-001/{p.name}')
              for p in sorted((ROOT/'data/runs/play-state-prospective-001').iterdir()) if p.is_dir()]
    out = []
    for split, video, folder in specs:
        run = ROOT/'data/runs'/folder
        gold = ROOT/'data/gold'
        if split == 'test':
            gold /= 'play-state-prospective-001'
        out.append({'split': split, 'video': video, 'frames': run/'frames.enriched.jsonl',
                    'baseline': run/'events.jsonl', 'gold': gold/f'{video}.hits_gold.csv',
                    'calibration': ROOT/'data/calibrations'/video/'v1.json'})
    return out


def load_case(spec):
    rows = jsonl(spec['frames'])
    calibration = json.loads(spec['calibration'].read_text(encoding='utf-8'))
    x, ids, raw = build_features(rows, calibration)
    dt = np.median(np.diff([r['timestamp_sec'] for r in rows]))
    fps = round(1/dt, 6)
    if not 29.8 <= fps <= 30.2:
        raise ValueError(f'Protocol expects native approximately 30 FPS, got {fps}')
    return {**spec, 'x': x, 'ids': ids, 'raw': raw, 'fps': fps, 'cal': calibration}


@torch.inference_mode()
def predict(model, x, device):
    model.eval()
    result = []
    for i in range(0, len(x), 512):
        result.append(model(torch.from_numpy(x[i:i+512]).to(device)).softmax(-1).cpu().numpy())
    return np.concatenate(result)


def macro_f1(y, pred):
    values = []
    for k in range(3):
        tp = int(((y == k) & (pred == k)).sum())
        den = int((y == k).sum()+(pred == k).sum())
        values.append(2*tp/den if den else 0)
    return float(np.mean(values))


def train(seed, train_cases, val_case, out, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    xs, ys = [], []
    for case in train_cases:
        y = window_labels(case['ids'], read_gold(case['gold']))
        xs.append(case['x'][y >= 0]); ys.append(y[y >= 0])
    x, y = np.concatenate(xs), np.concatenate(ys)
    vy = window_labels(val_case['ids'], read_gold(val_case['gold']))
    vx = val_case['x'][vy >= 0]; vy = vy[vy >= 0]
    counts = np.bincount(y, minlength=3)
    weights = len(y)/(3*np.maximum(counts, 1))
    model = HitNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=256,
                        shuffle=True, num_workers=0, generator=generator)
    best = -1
    history = []
    checkpoint = out/f'hitnet-local-seed-{seed}.pt'
    for epoch in range(1, 81):
        model.train()
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossfn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            total += float(loss.detach())*len(bx)
        vf = macro_f1(vy, predict(model, vx, device).argmax(-1))
        history.append({'epoch': epoch, 'loss': total/len(x), 'validation_window_macro_f1': vf})
        if vf > best:
            best = vf
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'seed': seed,
                        'validation_window_macro_f1': vf, 'origin': 'locally_trained_not_official'}, checkpoint)
        if epoch % 10 == 0 or epoch == 1:
            print(f'seed={seed} epoch={epoch}/80 val_window_macro_f1={vf:.4f} best={best:.4f}', flush=True)
    dump(out/f'training-seed-{seed}.json', {'class_counts': counts.tolist(), 'history': history})
    return checkpoint


def metrics(events, gold):
    return {name: evaluate_candidates(events, gold, tolerance=5, reaction_allowance_frames=extra)
            for name, extra in [('strict', 0), ('relaxed', 6)]}


def aggregate(reports, mode):
    rows = [r[mode] for r in reports]
    tp = sum(r['matched_count'] for r in rows)
    pred = sum(r['candidate_count'] for r in rows)
    gt = sum(r['gold_count'] for r in rows)
    p = tp/pred if pred else 0
    r = tp/gt if gt else 0
    matches = [m for row in rows for m in row['matches']]
    scored = [m for m in matches if m['gold_hitter'] != 'unknown']
    return {'precision': p, 'recall': r, 'f1': 2*p*r/(p+r) if p+r else 0,
            'predicted': pred, 'gold': gt, 'tp': tp, 'fp': pred-tp, 'fn': gt-tp,
            'hitter_accuracy': sum(m['predicted_hitter']==m['gold_hitter'] for m in scored)/len(scored) if scored else None,
            'median_absolute_frame_error': float(np.median([m['absolute_frame_error'] for m in matches])) if matches else None,
            'median_signed_frame_error': float(np.median([m['frame_error'] for m in matches])) if matches else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():
        raise ValueError('Use a NEW output directory inside this experiment; results are never overwritten')
    out.mkdir(parents=True)
    started = time.time()
    specs = cases()
    protected = sorted(set([p for s in specs for p in (s['frames'], s['baseline'], s['gold'], s['calibration'])]
                           + list((ROOT/'badminton_pipeline').rglob('*.py'))
                           + list((ROOT/'models/bst').glob('*.*'))))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    dump(out/'manifest-before.json', {'protected': before, 'protocol_sha256': sha(HERE/'PROTOCOL.md'),
                                    'source_hashes': {str(p.relative_to(HERE)): sha(p) for p in HERE.glob('*.py')},
                                    'upstream_hashes': {p.name:sha(p) for p in (HERE/'upstream').iterdir() if p.is_file()},
                                    'cases': [{k:str(v) for k,v in s.items()} for s in specs]})
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}; torch={torch.__version__}; preparing train/validation only', flush=True)
    loaded = [load_case(s) for s in specs if s['split'] != 'test']
    train_cases = [s for s in loaded if s['split'] == 'train']
    val_case = next(s for s in loaded if s['split'] == 'validation')
    checkpoints = [train(seed, train_cases, val_case, out, device) for seed in (17,29,43)]
    dump(out/'frozen-checkpoints.json', {p.name:sha(p) for p in checkpoints})
    print('All checkpoints frozen. Beginning retrospective six-clip evaluation.', flush=True)
    tests = [load_case(s) for s in specs if s['split'] == 'test']
    results = {}
    baseline_reports = []
    for case in tests:
        report = metrics(jsonl(case['baseline']), read_gold(case['gold']))
        baseline_reports.append(report)
        dump(out/'baseline'/f'{case["video"]}.json', report)
    results['baseline_rules_v06'] = {mode:aggregate(baseline_reports, mode) for mode in ('strict','relaxed')}
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model = HitNet().to(device)
        model.load_state_dict(state['state_dict'])
        for mode in ('dp', 'naive'):
            reports = []
            name = f'hitnet_seed_{state["seed"]}_{mode}'
            for case in tests:
                probs_file = out/f'seed-{state["seed"]}'/f'{case["video"]}.probabilities.npy'
                if probs_file.exists():
                    probs = np.load(probs_file)
                else:
                    probs = predict(model, case['x'], device)
                    probs_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(probs_file, probs)
                hits, sides, meta = decode(probs, case['fps'], case['raw'], case['cal'], mode)
                events = [{'video_id':case['video'], 'event_id':f'{name}_{i:04d}',
                           'candidate_frame':int(case['ids'][0])+h,
                           'predicted_hitter':'lower' if side==1 else 'upper',
                           'rally_id':'unknown', 'event_source':name}
                          for i,(h,side) in enumerate(zip(hits,sides))]
                folder = out/name/case['video']
                jsonl(folder/'events.jsonl', events)
                jsonl(folder/'bst-windows.jsonl', bst_windows(events, case['fps'], int(case['ids'][0]), int(case['ids'][-1])))
                report = metrics(events, read_gold(case['gold']))
                reports.append(report)
                dump(folder/'evaluation.json', {**report, 'decoder_metadata':meta})
            results[name] = {mode:aggregate(reports, mode) for mode in ('strict','relaxed')}
    changed = [name for name, digest in before.items() if sha(ROOT/name) != digest]
    summary = {'identity':'Local architecture reproduction, NOT official pretrained MonoTrack',
               'device':device, 'torch':torch.__version__, 'elapsed_sec':time.time()-started,
               'protected_files_checked':len(before), 'protected_files_changed':changed,
               'checkpoint_hashes':{p.name:sha(p) for p in checkpoints}, 'results':results}
    dump(out/'summary.json', summary)
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
