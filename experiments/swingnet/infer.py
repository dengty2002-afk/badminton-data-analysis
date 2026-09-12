"""Standalone pretrained SwingNet and frozen fusion inference; no Gold or UI."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from swingnet import HERE,load_model,preprocess_video,infer_video,author_events,fuse
from run_experiment import ROOT,dump,jsonl,sha
from badminton_pipeline.tracking.postprocess import score_hit_candidates


def window_rows(events,fps,n):
    t=fps//2;extension=t//2;limit=fps*3//2
    result=[]
    for i,e in enumerate(events):
        f=e['candidate_frame']
        prev=events[i-1]['candidate_frame'] if i else None
        nxt=events[i+1]['candidate_frame'] if i+1<len(events) else None
        if prev is not None and f-prev>6*fps:prev=None
        if nxt is not None and nxt-f>6*fps:nxt=None
        result.append({**e,'start_frame':int(max(0,f-limit,prev if prev is not None else f-t)),
                       'end_frame_exclusive':int(min(n,f+limit+extension,nxt+extension if nxt is not None else f+t)),
                       'window_policy':'BST upstream previous/next contact with maximum limits',
                       'rally_policy':'automatic six-second gap, no Gold'})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--video-id',required=True)
    p.add_argument('--frames',type=Path)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--selection',type=Path,default=HERE/'results/frozen-selection.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(HERE):p.error('output must be new and inside this experiment')
    if bool(args.frames)!=bool(args.baseline):p.error('frames and baseline are supplied together for fusion')
    out.mkdir(parents=True)
    checkpoint=HERE/'assets/weights/SwingNet/models/fold5_swingnet_3000.pth.tar'
    if sha(checkpoint)!='d896c3e4f1a9a8ad629feadc9d5c29944d9cbb00ddcb86b341bb68e6b8e8901d':p.error('Wrong checkpoint')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model(checkpoint,device);start=time.perf_counter()
    info=preprocess_video(args.video,out/'motion-xgg.mp4')
    probs=infer_video(model,out/'motion-xgg.mp4',device)
    hits=author_events(probs)
    events=[{'video_id':args.video_id,'candidate_frame':f,'predicted_hitter':'unknown','event_id':f'swing_{i:05d}'} for i,f in enumerate(hits)]
    jsonl(out/'swingnet-events.jsonl',events)
    jsonl(out/'swingnet-bst-windows.jsonl',window_rows(events,info['fps'],len(probs)))
    selected=events;selection_note='SwingNet only; experimental'
    if args.frames:
        rows=jsonl(args.frames)
        if any(r['video_id']!=args.video_id or r['frame_idx']!=i for i,r in enumerate(rows)) or len(rows)!=len(probs):
            p.error('features must match the complete source video and video ID')
        base=jsonl(args.baseline)
        physical=score_hit_candidates(rows,fps=info['fps'],threshold=0,min_separation_frames=1)
        selection=json.loads(args.selection.read_text(encoding='utf8'))
        fused=fuse(base,hits,probs,physical,selection['selected_fusion'])
        fused=[{**e,'video_id':args.video_id,'event_id':f'fusion_{i:05d}'} for i,e in enumerate(fused)]
        jsonl(out/'fusion-events.jsonl',fused)
        jsonl(out/'fusion-bst-windows.jsonl',window_rows(fused,info['fps'],len(probs)))
        selected=fused if selection['validation_gate_passed'] else base
        selection_note=selection['deployment_choice']
    jsonl(out/'selected-events.jsonl',selected)
    np.save(out/'probabilities.npy',probs)
    report={**info,'video_id':args.video_id,'checkpoint_sha256':sha(checkpoint),'source_sha256':sha(args.video),
            'swingnet_count':len(events),'selected_count':len(selected),'selection':selection_note,
            'elapsed_sec':time.perf_counter()-start,'uses_gold':False,'device':device}
    dump(out/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
