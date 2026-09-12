import ast
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch
from swingnet import HERE,SwingNet,load_model,magnitude,motion_frame,tensor_frame,author_events,percentile_ranks,fuse


class ContractTests(unittest.TestCase):
    def test_checkpoint_strict_load(self):
        model=load_model(HERE/'assets/weights/SwingNet/models/fold5_swingnet_3000.pth.tar','cpu')
        self.assertEqual(len(model.state_dict()),322)
        with torch.no_grad():self.assertEqual(model(torch.zeros(1,2,3,250,180)).shape,(2,2))

    def test_motion_equals_historical_code(self):
        source=(HERE/'upstream/rt_conversion_historical.py').read_text(encoding='utf8')
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='imgradient')
        ns={'cv2':cv2,'np':np}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream','exec'),ns)
        rng=np.random.default_rng(42)
        a=rng.integers(0,255,(72,128,3),dtype=np.uint8)
        b=rng.integers(0,255,(72,128,3),dtype=np.uint8)
        np.testing.assert_allclose(magnitude(a),ns['imgradient'](cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)))
        begin=source.index('        delta_frame = 0.5 * np.abs(mgrad2 - mgrad0)')
        end=source.index('        # display',begin)
        import textwrap
        ns.update(mgrad0=magnitude(a),mgrad2=magnitude(b),frameRGB1=a)
        exec(textwrap.dedent(source[begin:end]),ns)
        np.testing.assert_array_equal(motion_frame(magnitude(a),magnitude(b)),ns['ndfrgb'])

    def test_static_frame_safe(self):
        self.assertEqual(int(motion_frame(np.zeros((9,9)),np.zeros((9,9))).sum()),0)

    def test_notebook_geometry(self):
        self.assertEqual(tensor_frame(np.zeros((720,1280,3),np.uint8)).shape,(3,250,180))

    def test_author_decoder_tail_behavior(self):
        probs=np.tile([0.,1.],(50,1));probs[5:9]=[.9,.1];probs[30:33]=[.9,.1]
        self.assertEqual(author_events(probs),[5])

    def test_ranks_ties(self):
        np.testing.assert_allclose(percentile_ranks([1,1,2]),[.25,.25,1])

    def test_fusion_adds_independent_candidate(self):
        p=np.tile([.1,.9],(60,1));p[40]=[.9,.1]
        events=fuse([], [40],p,[{'candidate_frame':40,'event_confidence':.6,'predicted_hitter':'upper'}],
                    {'keep_rank':.4,'add_evidence':.35,'refine':True})
        self.assertEqual(events[0]['candidate_frame'],40)
        self.assertEqual(events[0]['predicted_hitter'],'upper')


if __name__=='__main__':unittest.main()
