"""Small synthetic matching example; no video, model or human data required."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from badminton_pipeline.evaluation.hit_events import read_gold,evaluate_candidates


def main():
    candidates=[{'video_id':'synthetic','event_id':'hit_1','candidate_frame':101},
                {'video_id':'synthetic','event_id':'hit_2','candidate_frame':350}]
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'gold.csv'
        path.write_text('video_id,rally_id,hit_frame,hitter,confidence,uncertainty_frames,notes\nsynthetic,rally_1,100,unknown,high,0,synthetic\nsynthetic,rally_1,200,unknown,high,0,synthetic\n',encoding='utf8')
        result=evaluate_candidates(candidates,read_gold(path),tolerance=5)
    assert result['matched_count']==1 and result['false_positive_count']==1 and result['false_negative_count']==1
    print(json.dumps({key:result[key] for key in ('precision','recall','f1','matched_count','false_positive_count','false_negative_count')},indent=2))


if __name__=='__main__':main()
