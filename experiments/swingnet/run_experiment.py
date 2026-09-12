"""Pretrained SwingNet development/validation selection and frozen regression comparison."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from badminton_pipeline.evaluation.hit_events import read_gold,evaluate_candidates
from badminton_pipeline.tracking.postprocess import score_hit_candidates
from swingnet import load_model,preprocess_video,infer_video,author_events,fuse


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def jsonl(path,rows=None):
    if rows is None:return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s]
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')


def specs():
    cases=[('development','vid_2e2258f261aa','full-local-v0.3-play-gate-dev'),
           ('development','vid_a598228fbc11','holdout-a598-play-gate-dev'),
           ('validation','vid_576b0c1ab370','validation-576b-play-gate-dev')]
    cases += [('regression',p.name,f'play-state-prospective-001/{p.name}')
              for p in sorted((ROOT/'data/runs/play-state-prospective-001').iterdir()) if p.is_dir()]
    result=[]
    for split,video,folder in cases:
        registration=ROOT/'data/videos'/f'{video}.json'
        reg=json.loads(registration.read_text(encoding='utf8'))
        run=ROOT/'data/runs'/folder
        gp=ROOT/'data/gold'
        if split=='regression':gp/='play-state-prospective-001'
        result.append({'split':split,'video':video,'source':Path(reg['path']),
                       'registration':registration,'frames':run/'frames.enriched.jsonl',
                       'baseline':run/'events.jsonl','gold':gp/f'{video}.hits_gold.csv'})
    return result


def prepare(spec,out,model,device):
    folder=out/spec['split']/spec['video'];folder.mkdir(parents=True,exist_ok=True)
    video=folder/'motion-xgg.mp4'
    start=time.perf_counter()
    info=preprocess_video(spec['source'],video)
    prep=time.perf_counter()-start
    start=time.perf_counter();probs=infer_video(model,video,device);infer=time.perf_counter()-start
    if len(probs)!=info['frames']:raise ValueError('Inference frame alignment changed')
    frames=jsonl(spec['frames'])
    if len(frames)!=len(probs) or any(r['frame_idx']!=i for i,r in enumerate(frames)):
        raise ValueError('Cached features do not correspond to full zero-based source video')
    physical=score_hit_candidates(frames,fps=info['fps'],threshold=0,min_separation_frames=1)
    baseline=jsonl(spec['baseline'])
    sf=author_events(probs)
    events=[{'candidate_frame':f,'predicted_hitter':'unknown','video_id':spec['video'],
             'event_id':f'swingnet_{i:05d}','event_source':'author_pretrained_swingnet'} for i,f in enumerate(sf)]
    np.save(folder/'probabilities.npy',probs)
    jsonl(folder/'swingnet-events.jsonl',events)
    jsonl(folder/'physical-proposals.jsonl',physical)
    dump(folder/'inference.json',{**info,'preprocess_seconds':prep,'inference_seconds':infer,
                                 'event_count':len(events),'motion_sha256':sha(video),
                                 'probability_min':float(probs[:,0].min()),'probability_max':float(probs[:,0].max()),
                                 'mean_hit_probability':float(probs[:,0].mean())})
    print(f'{spec["split"]} {spec["video"]}: {len(probs)} frames, {len(events)} SwingNet events, preprocess={prep:.1f}s infer={infer:.1f}s',flush=True)
    return {**spec,'folder':folder,'probs':probs,'physical':physical,'baseline_rows':baseline,
            'swing_frames':sf,'swing_events':events,'fps':info['fps'],'frame_count':len(probs)}


def score(events,gold):
    return {mode:evaluate_candidates(events,gold,tolerance=5,reaction_allowance_frames=extra)
            for mode,extra in [('strict',0),('relaxed',6)]}


def aggregate(reports,mode='strict'):
    rows=[r[mode] for r in reports]
    tp=sum(r['matched_count'] for r in rows);pred=sum(r['candidate_count'] for r in rows);gold=sum(r['gold_count'] for r in rows)
    p=tp/pred if pred else 0.;r=tp/gold if gold else 0.
    matches=[m for row in rows for m in row['matches']]
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.,'predicted':pred,'gold':gold,
            'tp':tp,'fp':pred-tp,'fn':gold-tp,
            'median_absolute_frame_error':float(np.median([m['absolute_frame_error'] for m in matches])) if matches else None,
            'median_signed_frame_error':float(np.median([m['frame_error'] for m in matches])) if matches else None}


def fused(case,config):
    rows=fuse(case['baseline_rows'],case['swing_frames'],case['probs'],case['physical'],config)
    return [{**row,'video_id':case['video'],'event_id':f'fusion_{i:05d}','event_source':'experimental_swingnet_rule_fusion'} for i,row in enumerate(rows)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(HERE):parser.error('Choose a NEW result directory inside this experiment')
    out.mkdir(parents=True)
    cases=specs()
    checkpoint=HERE/'assets/weights/SwingNet/models/fold5_swingnet_3000.pth.tar'
    if sha(checkpoint)!='d896c3e4f1a9a8ad629feadc9d5c29944d9cbb00ddcb86b341bb68e6b8e8901d':raise ValueError('Wrong checkpoint')
    protected=set(p for s in cases for p in [s['source'],s['registration'],s['frames'],s['baseline'],s['gold']])
    protected.update((ROOT/'badminton_pipeline').rglob('*.py'))
    protected.update(p for p in (ROOT/'models').rglob('*') if p.is_file())
    before={str(p.resolve()):sha(p) for p in sorted(protected)}
    dump(out/'manifest-before.json',{'protected':before,'checkpoint_sha256':sha(checkpoint),
         'experiment_sources':{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*')
                               if p.is_file() and p.suffix in ('.py','.md','.ipynb') and not p.is_relative_to(out)}})
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model(checkpoint,device)
    dump(out/'model-load.json',{'strict_state_dict_load':True,'state_tensors':len(model.state_dict()),
                               'parameters':sum(p.numel() for p in model.parameters()),'device':device,'torch':torch.__version__})
    print('All pretrained tensors loaded strictly; running development/validation.',flush=True)
    dev=[prepare(s,out,model,device) for s in cases if s['split']=='development']
    val=prepare(next(s for s in cases if s['split']=='validation'),out,model,device)
    for case in dev+[val]:case['gold_rows']=read_gold(case['gold'])
    candidates=[]
    for keep,add,refine in itertools.product([0.,.4,.6],[0.,.2,.35],[False,True]):
        config={'keep_rank':keep,'add_evidence':add,'refine':refine}
        reports=[score(fused(c,config),c['gold_rows']) for c in dev]
        candidates.append({'config':config,'development':aggregate(reports)})
    candidates.sort(key=lambda r:(r['development']['f1'],r['development']['precision'],-r['development']['predicted']),reverse=True)
    dump(out/'development-search.json',candidates)
    top=candidates[:3]
    for item in top:item['validation']=aggregate([score(fused(val,item['config']),val['gold_rows'])])
    top.sort(key=lambda r:(r['validation']['f1'],r['validation']['precision'],-r['validation']['predicted']),reverse=True)
    best=top[0]
    base_val=aggregate([score(val['baseline_rows'],val['gold_rows'])])
    passed=best['validation']['f1']>base_val['f1'] and best['validation']['recall']>=base_val['recall']-.02
    selection={'top3':top,'baseline_validation':base_val,'selected_fusion':best['config'],
               'validation_gate_passed':passed,'deployment_choice':'fusion' if passed else 'keep_baseline',
               'note':'Frozen before regression scoring; no independent blind-test claim.'}
    dump(out/'frozen-selection.json',selection)
    selection_hash=sha(out/'frozen-selection.json')
    print('Frozen selection: '+json.dumps(selection,ensure_ascii=False),flush=True)
    results={}
    regression=[prepare(s,out,model,device) for s in cases if s['split']=='regression']
    for split,group in [('development',dev),('validation',[val]),('regression',regression)]:
        bymethod={'rules':[],'swingnet':[],'fusion':[]}
        for case in group:
            gold=read_gold(case['gold'])
            for name,events in [('rules',case['baseline_rows']),('swingnet',case['swing_events']),('fusion',fused(case,best['config']))]:
                report=score(events,gold)
                bymethod[name].append(report)
                dump(case['folder']/f'{name}-evaluation.json',report)
                if name=='fusion':jsonl(case['folder']/'fusion-events.jsonl',events)
        results[split]={name:{mode:aggregate(reports,mode) for mode in ('strict','relaxed')} for name,reports in bymethod.items()}
    changed=[p for p,h in before.items() if sha(p)!=h]
    summary={'identity':'Author pretrained badminton SwingNet with historical author motion transform',
             'selection_sha256':selection_hash,'deployment_choice':selection['deployment_choice'],
             'protected_files_checked':len(before),'protected_files_changed':changed,'results':results}
    dump(out/'summary.json',summary)
    if changed:raise RuntimeError(f'Protected files changed: {changed}')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
