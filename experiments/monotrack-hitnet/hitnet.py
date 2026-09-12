"""Isolated MonoTrack architecture reproduction; see PROTOCOL.md and upstream/LICENSE."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn


class HitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(78, 64, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(128, 3)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return self.classifier(x.max(dim=1).values)


def scale_features(x):
    x = np.asarray(x, dtype=np.float32).copy()
    for parity in (0, 1):
        values = x[:, parity::2]
        valid = np.abs(values) >= 1e-6
        if valid.any():
            low, high = values[valid].min(), values[valid].max()
            values[valid] = (values[valid] - low) / max(float(high-low), 1e-6) + 1
    return x


def build_features(frames, calibration):
    ids = np.array([f['frame_idx'] for f in frames], dtype=np.int64)
    if not np.all(np.diff(ids) == 1):
        raise ValueError('HitNet input must be contiguous; never bridge edited video gaps')
    # Local calibration TL,TR,BR,BL -> official Court TL,TR,BL,BR;
    # official detect_hits then indexes [1,2,0,3].
    corners = np.array([[c['x'], c['y']] for c in calibration['corners']], np.float32)
    corners = corners[[1, 3, 0, 2]].flatten()
    data = np.zeros((len(frames), 78), dtype=np.float32)
    data[:, 70:] = corners
    for i, frame in enumerate(frames):
        point = frame.get('shuttle', {}).get('smoothed_xy')
        if isinstance(point, list) and len(point) == 2:
            data[i, :2] = point
        for side, start in [('lower', 2), ('upper', 36)]:
            person = next((p for p in frame.get('players', []) if p['identity'] == side), None)
            if person:
                points = np.array(person['keypoints_xy'], dtype=np.float32)
                valid = ~np.array(person.get('keypoint_missing', [False]*17), dtype=bool)
                points[~valid] = 0
                data[i, start:start+34] = points.flatten()
    data[~np.isfinite(data)] = 0
    # Scale only points represented in the windows, equivalent to scale_data
    # applied after stacking all 12 offsets in the original implementation.
    scaled = scale_features(data)
    windows = np.stack([scaled[i:i+12] for i in range(len(data)-11)])
    return windows, ids, data


def window_labels(ids, gold):
    labels = np.zeros(len(ids), dtype=np.int64)
    offset = int(ids[0])
    for hit in gold:
        if hit.hitter not in ('lower', 'upper'):
            raise ValueError('Unknown hitter requires an explicit masked-label policy')
        index = hit.hit_frame - offset
        if not 0 <= index < len(labels):
            raise ValueError('Gold outside feature frame range')
        labels[index] = 1 if hit.hitter == 'lower' else 2
    out = []
    for i in range(len(ids)-11):
        found = set(labels[i+6:i+12]) - {0}
        out.append(-1 if len(found) > 1 else next(iter(found), 0))
    return np.array(out, dtype=np.int64)


def upstream_method(name):
    source = Path(__file__).parent/'upstream'/'hit_detector.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLHitDetector')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    env = {'np': np}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), env)
    return env[name]


def decode(probabilities, fps, raw, calibration, mode='dp'):
    metadata = {'decoder': mode, 'temperature': 1.0}
    obj = SimpleNamespace(fps=fps)
    if mode == 'dp':
        if not np.any(probabilities[:, 0] < 0.1):
            metadata['undefined_tau_no_confident_hits'] = True
            return [], [], metadata
        hits, sides = upstream_method('dp_postprocessing')(obj, probabilities)
    else:
        c = np.array([[c['x'], c['y']] for c in calibration['corners']])
        obj.court = SimpleNamespace(corners=c[[0,1,3,2]])
        obj.trajectory = SimpleNamespace(X=raw[:,0], Y=raw[:,1])
        obj.model = SimpleNamespace(input_shape=(None, 936))
        hits, sides = upstream_method('naive_postprocessing')(obj, probabilities)
    pairs = [(int(h), int(s)) for h,s in zip(hits, sides) if 0 <= h < len(raw) and s in (1,2)]
    return [p[0] for p in pairs], [p[1] for p in pairs], metadata


def bst_windows(events, fps, first_frame, last_frame):
    # Separate experimental artifact, no production adapter changes.
    t, limit = fps//2, fps*3//2
    eps = t//2
    result = []
    for i, event in enumerate(events):
        current = event['candidate_frame']
        prev = events[i-1]['candidate_frame'] if i else None
        nxt = events[i+1]['candidate_frame'] if i+1 < len(events) else None
        # Six-second gap is an explicit automatic boundary heuristic, never Gold.
        if prev is not None and current-prev > 6*fps:
            prev = None
        if nxt is not None and nxt-current > 6*fps:
            nxt = None
        start = max(first_frame, current-limit, prev if prev is not None else current-t)
        end = min(last_frame+1, current+limit+eps, nxt+eps if nxt is not None else current+t)
        result.append({**event, 'start_frame': int(start), 'end_frame_exclusive': int(end),
                       'window_policy': 'upstream_between_2_hits_with_max_limits',
                       'rally_boundary_policy': 'automatic_6_second_gap'})
    return result
