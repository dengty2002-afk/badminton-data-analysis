"""End-to-end CLI check using a 64-frame derivative of a development video."""
import json
from pathlib import Path
import subprocess
import sys
import cv2
from run_experiment import HERE, specs, jsonl, dump


def main():
    spec=next(s for s in specs() if s['split']=='development')
    source=spec['source']
    target=HERE/'smoke-input.mp4'
    out=HERE/'smoke-result'
    if '--fusion-only' not in sys.argv:
        if target.exists() or out.exists():raise RuntimeError('Smoke outputs already exist')
        cap=cv2.VideoCapture(str(source))
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS),(1280,720))
        assert writer.isOpened()
        for _ in range(64):
            ok,frame=cap.read();assert ok
            writer.write(frame)
        cap.release();writer.release()
        subprocess.run([sys.executable,'-B',str(HERE/'infer.py'),'--video',str(target),
                        '--video-id','smoke_development_64','--output',str(out)],check=True)
    result=json.loads((out/'summary.json').read_text(encoding='utf8'))
    assert result['frames']==64 and result['uses_gold'] is False
    for name in ('swingnet-events.jsonl','selected-events.jsonl'):
        rows=[json.loads(s) for s in (out/name).read_text(encoding='utf8').splitlines()]
        assert all(0<=e['candidate_frame']<64 for e in rows)
    print('CLI smoke passed: 64 frames, no Gold, aligned event indices.')
    fusion_out=HERE/'smoke-fusion-result'
    fixture=HERE/'smoke-fusion-input'
    if fixture.exists() or fusion_out.exists():raise RuntimeError('Fusion smoke outputs already exist')
    fixture.mkdir()
    jsonl(fixture/'frames.jsonl',[{**r,'video_id':'smoke_development_64'} for r in jsonl(spec['frames'])[:64]])
    jsonl(fixture/'baseline.jsonl',[{**r,'video_id':'smoke_development_64'} for r in jsonl(spec['baseline']) if r['candidate_frame']<64])
    subprocess.run([sys.executable,'-B',str(HERE/'infer.py'),'--video',str(target),
                    '--video-id','smoke_development_64','--output',str(fusion_out),
                    '--frames',str(fixture/'frames.jsonl'),'--baseline',str(fixture/'baseline.jsonl')],check=True)
    selection=json.loads((HERE/'results/frozen-selection.json').read_text(encoding='utf8'))
    expected=jsonl(fusion_out/'fusion-events.jsonl') if selection['validation_gate_passed'] else jsonl(fixture/'baseline.jsonl')
    assert jsonl(fusion_out/'selected-events.jsonl')==expected
    for w in jsonl(fusion_out/'fusion-bst-windows.jsonl'):
        assert 0<=w['start_frame']<=w['candidate_frame']<w['end_frame_exclusive']<=64
    dump(fusion_out/'checks.json',{'selected_events_follow_frozen_gate':True,'window_bounds_valid':True,'uses_gold':False})
    print('Fusion CLI smoke passed: frozen selection and BST window bounds.')


if __name__=='__main__':main()
