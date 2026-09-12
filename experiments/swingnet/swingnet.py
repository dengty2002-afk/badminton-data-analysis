"""Pretrained badminton SwingNet adapter, pinned author preprocessing and model tensors."""
from pathlib import Path
import importlib.util
import cv2
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


class SwingNet(nn.Module):
    def __init__(self):
        super().__init__()
        spec=importlib.util.spec_from_file_location('swingnet_upstream_mobilenet',HERE/'upstream/golfdb_MobileNetV2.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        net=module.MobileNetV2(width_mult=1.)
        self.cnn=nn.Sequential(*list(net.children())[0][:19])
        self.rnn=nn.LSTM(1280,256,1,batch_first=True,bidirectional=True)
        self.lin=nn.Linear(512,2)

    def forward(self,x):
        batch,steps,c,h,w=x.shape
        x=self.cnn(x.reshape(batch*steps,c,h,w)).mean(3).mean(2)
        x,_=self.rnn(x.reshape(batch,steps,1280))
        return self.lin(x).reshape(batch*steps,2)


def load_model(path,device):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=SwingNet()
    model.load_state_dict(state['model_state_dict'],strict=True)
    return model.to(device).eval()


def magnitude(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0)
    gy=cv2.Sobel(gray,cv2.CV_64F,0,1)
    return np.sqrt(gx**2+gy**2)


def motion_frame(old,current):
    local=cv2.boxFilter(0.5*np.abs(current-old),-1,(3,3),normalize=True)
    low,high=float(local.min()),float(local.max())
    signal=np.zeros(local.shape,np.uint8) if high==low else np.uint8(np.round(255*(local-low)/(high-low)))
    return np.stack([np.zeros_like(signal),signal,signal],axis=-1)


def preprocess_video(source,destination):
    cap=cv2.VideoCapture(str(source))
    fps=cap.get(cv2.CAP_PROP_FPS)
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w,h)!=(1280,720): raise ValueError('Upstream notebook requires 1280x720 input')
    destination.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not cap.isOpened() or not writer.isOpened(): raise RuntimeError('Video reader/writer unavailable')
    history=[]; n=0
    while True:
        ret,frame=cap.read()
        if not ret: break
        mag=magnitude(frame)
        if n==0: transformed=np.zeros_like(frame)
        else: transformed=motion_frame(history[0],mag)
        writer.write(transformed)
        history.append(mag)
        if len(history)>2: history.pop(0)
        n+=1
    cap.release();writer.release()
    if n!=expected: raise RuntimeError(f'Frame count mismatch: {n} != {expected}')
    return {'frames':n,'fps':fps,'width':w,'height':h}


def tensor_frame(frame):
    resized=cv2.resize(frame[:,280:1000],(180,180),interpolation=cv2.INTER_AREA)
    padded=cv2.copyMakeBorder(resized,35,35,0,0,cv2.BORDER_CONSTANT,value=[.406*255,.456*255,.485*255])
    rgb=cv2.cvtColor(padded,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return rgb.transpose(2,0,1)


@torch.inference_mode()
def infer_video(model,video,device):
    cap=cv2.VideoCapture(str(video)); batch=[]; output=[]
    while True:
        ret,frame=cap.read()
        if ret: batch.append(tensor_frame(frame))
        if len(batch)==64 or (not ret and batch):
            x=torch.from_numpy(np.stack(batch)[None]).to(device)
            output.append(model(x).softmax(-1).cpu().numpy())
            batch=[]
        if not ret:break
    cap.release()
    if not output: raise RuntimeError('Empty processed video')
    return np.concatenate(output)


def author_events(probs):
    """Literal notebook grouping behavior, including omission of final candidate."""
    p=probs[:,0]
    threshold=np.quantile(p,.8,method='higher')
    points=np.flatnonzero(p>threshold).tolist()
    clusters=[[]]
    for i in range(len(points)-1):
        clusters[-1].append(points[i])
        if points[i+1]-points[i]!=1: clusters.append([])
    return [int(g[int(np.argmax(p[g]))]) for g in clusters if len(g)>=3]


def percentile_ranks(values):
    # Average tie ranks, expressed in [0,1].
    unique,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    ends=np.cumsum(counts); starts=ends-counts
    return ((starts+ends-1)/2/max(1,len(values)-1))[inverse]


def fuse(baseline,swing_frames,probs,physical,config):
    ranks=percentile_ranks(probs[:,0]); n=len(ranks)
    def sr(f): return float(ranks[max(0,f-4):min(n,f+5)].max())
    def nearby(f): return [p for p in physical if abs(p['candidate_frame']-f)<=4]
    def evidence(f): return max([p['event_confidence'] for p in nearby(f)],default=0.)
    choices=[]
    for e in baseline:
        f=e['candidate_frame']
        if sr(f)>=config['keep_rank']:
            choices.append({**e,'sources':['rule'],'fusion_score':sr(f)+evidence(f)})
    for f in swing_frames:
        if evidence(f)>=config['add_evidence']:
            close=nearby(f)
            best=max(close,key=lambda p:p['event_confidence']) if close else {}
            choices.append({'candidate_frame':f,'predicted_hitter':best.get('predicted_hitter','unknown'),
                            'sources':['swingnet'],'fusion_score':sr(f)+evidence(f)})
    # Group within six frames of first member; do not chain unrelated events.
    groups=[]
    for e in sorted(choices,key=lambda e:e['candidate_frame']):
        if not groups or e['candidate_frame']-groups[-1][0]['candidate_frame']>6:groups.append([])
        groups[-1].append(e)
    output=[]
    for group in groups:
        best=max(group,key=lambda e:(e['fusion_score'],'rule' in e['sources']))
        selected={**best,'sources':sorted(set(s for e in group for s in e['sources']))}
        f=selected['candidate_frame']
        if config['refine']:
            peaks=nearby(f)
            if peaks:
                peak=max(peaks,key=lambda e:e['event_confidence']*(.5+.5*sr(e['candidate_frame'])))
                if peak['event_confidence']>=.35:
                    selected['candidate_frame']=peak['candidate_frame']
                    selected['predicted_hitter']=peak.get('predicted_hitter','unknown')
        output.append(selected)
    # Refinement can align two groups; ensure one event per source frame.
    unique={}
    for e in output:
        f=e['candidate_frame']
        if f not in unique or e['fusion_score']>unique[f]['fusion_score']:unique[f]=e
    return sorted(unique.values(),key=lambda e:e['candidate_frame'])
