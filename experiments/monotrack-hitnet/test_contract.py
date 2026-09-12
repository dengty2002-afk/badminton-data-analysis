import unittest
from types import SimpleNamespace
import numpy as np
from hitnet import HitNet, build_features, window_labels, scale_features, bst_windows, decode
import torch


class ContractTests(unittest.TestCase):
    def test_network_shape(self):
        self.assertEqual(tuple(HitNet()(torch.zeros(3,12,78)).shape), (3,3))

    def test_label_offset(self):
        ids = np.arange(100,124)
        gold = [SimpleNamespace(hit_frame=111,hitter='lower')]
        y = window_labels(ids,gold)
        self.assertEqual(np.flatnonzero(y).tolist(), list(range(6)))

    def test_missing_and_scaling(self):
        x = np.array([[0,0,2,5],[4,10,6,15]], np.float32)
        y = scale_features(x)
        self.assertEqual(y[0,0], 0)
        np.testing.assert_allclose(y[0,2:], [1,1])
        np.testing.assert_allclose(y[1,2:], [2,2])

    def test_feature_order(self):
        cal = {'corners':[{'x':x,'y':y} for x,y in [(10,10),(20,10),(20,20),(10,20)]]}
        frames = [{'frame_idx':i,'shuttle':{'smoothed_xy':[3,4]},'players':[
            {'identity':'lower','keypoints_xy':[[5,6]]*17},
            {'identity':'upper','keypoints_xy':[[7,8]]*17}]} for i in range(13)]
        x, ids, raw = build_features(frames,cal)
        self.assertEqual(x.shape,(2,12,78))
        np.testing.assert_array_equal(raw[0,:6], [3,4,5,6,5,6])
        np.testing.assert_array_equal(raw[0,36:38],[7,8])
        np.testing.assert_array_equal(raw[0,70:],[20,10,10,20,10,10,20,20])
        frames[3]['frame_idx'] = 100
        with self.assertRaises(ValueError): build_features(frames,cal)

    def test_window_matches_upstream_example(self):
        e = [{'candidate_frame':f} for f in [30,60,90]]
        windows = bst_windows(e,30,0,200)
        self.assertEqual((windows[1]['start_frame'],windows[1]['end_frame_exclusive']),(30,97))

    def test_dp_no_confident_guard(self):
        p = np.tile([.9,.05,.05], (20,1))
        hits, sides, meta = decode(p,30,np.zeros((31,78)),{},'dp')
        self.assertFalse(hits)
        self.assertTrue(meta['undefined_tau_no_confident_hits'])

    def test_dp_alternation_and_offset(self):
        p = np.tile([.999,.0005,.0005],(100,1))
        p[20:26] = [.01,.98,.01]
        p[60:66] = [.01,.01,.98]
        hits, sides, _ = decode(p,30,np.zeros((111,78)),{},'dp')
        self.assertEqual(sides,[1,2])
        self.assertTrue(28 <= hits[0] <= 34)
        self.assertTrue(68 <= hits[1] <= 74)


if __name__ == '__main__':
    unittest.main()
