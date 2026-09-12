"""Read-only structural audit of frozen hit evaluations. Does not run or tune BST."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
RESULTS=HERE.parent/'results'


def main():
    protected={}
    def read(path):
        raw=path.read_bytes()
        protected[str(path.resolve())]=hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8-sig')
    summary=json.loads(read(RESULTS/'summary.json'))
    answer={'scope':'Six existing retrospective regression clips; no new inference, tuning or stroke labels.',
            'structure_definition':'Within a Gold rally, every hit in a pair/triple is matched and the corresponding candidates are consecutive in the full predicted timeline. Ignores player identity, stroke class and exact BST window bounds.',
            'strict':{},'relaxed':{}}
    for mode in ('strict','relaxed'):
        for method in ('rules','swingnet','fusion'):
            total={'gold_pairs':0,'both_detected':0,'clean_adjacent_pairs':0,
                   'gold_triples':0,'all_three_detected':0,'clean_adjacent_triples':0,
                   'gold_upper':0,'gold_lower':0,'matched_upper':0,'matched_lower':0}
            per_video=[]
            for folder in sorted((RESULTS/'regression').iterdir()):
                report=json.loads(read(folder/f'{method}-evaluation.json'))[mode]
                gold=list(csv.DictReader(read(PROJECT/'data/gold/play-state-prospective-001'/f'{folder.name}.hits_gold.csv').splitlines()))
                frames=sorted([r['candidate_frame'] for r in report['matches']]+[r['candidate_frame'] for r in report['false_positives']])
                assert len(frames)==len(set(frames))==report['candidate_count']
                index={f:i for i,f in enumerate(frames)}
                matched={r['gold_frame']:index[r['candidate_frame']] for r in report['matches']}
                assert len(matched)==report['matched_count']
                groups={};counts={k:0 for k in total}
                for row in gold:
                    frame=int(row['hit_frame']);groups.setdefault(row['rally_id'],[]).append(frame)
                    if row['hitter'] in ('upper','lower'):
                        counts['gold_'+row['hitter']]+=1
                        counts['matched_'+row['hitter']]+=int(frame in matched)
                for fs in groups.values():
                    fs.sort()
                    for size,den,num,clean in [(2,'gold_pairs','both_detected','clean_adjacent_pairs'),
                                               (3,'gold_triples','all_three_detected','clean_adjacent_triples')]:
                        for i in range(len(fs)-size+1):
                            window=fs[i:i+size];counts[den]+=1
                            if all(f in matched for f in window):
                                counts[num]+=1
                                counts[clean]+=int(all(matched[b]==matched[a]+1 for a,b in zip(window,window[1:])))
                assert counts['clean_adjacent_pairs']<=counts['both_detected']<=counts['gold_pairs']
                assert counts['clean_adjacent_triples']<=counts['all_three_detected']<=counts['gold_triples']
                per_video.append({'video_id':folder.name,**counts})
                total={k:total[k]+counts[k] for k in total}
            metrics=summary['results']['regression'][method][mode]
            assert total['matched_upper']+total['matched_lower']==metrics['tp']
            answer[mode][method]={**total,'per_video':per_video,
                'clean_pair_fraction':total['clean_adjacent_pairs']/total['gold_pairs'],
                'clean_triple_fraction':total['clean_adjacent_triples']/total['gold_triples'],
                'upper_detection_recall':total['matched_upper']/total['gold_upper'],
                'lower_detection_recall':total['matched_lower']/total['gold_lower'],
                'event_count_ratio':metrics['predicted']/metrics['gold']}
    answer['hypothetical_bst_example']={'conditional_type_accuracy_assumed_not_measured':.85,
        'assumptions':'Frozen strict matching; classify every fusion candidate once; no rejection, additional detection or side requirement.',
        'typed_micro_precision':summary['results']['regression']['fusion']['strict']['precision']*.85,
        'typed_micro_recall':summary['results']['regression']['fusion']['strict']['recall']*.85}
    answer['protected_inputs_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    assert answer['protected_inputs_unchanged']
    (HERE/'metrics.json').write_text(json.dumps(answer,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (HERE/'input-manifest.json').write_text(json.dumps(protected,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({mode:{name:{k:v for k,v in data.items() if k!='per_video'} for name,data in answer[mode].items()} for mode in ('strict','relaxed')},indent=2))


if __name__=='__main__':main()
