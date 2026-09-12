import unittest

from badminton_pipeline.tracking.play_state_gate import apply_play_state_gate


def frame(index, x):
    return {
        "frame_idx": index,
        "width": 100,
        "height": 100,
        "shuttle": {"smoothed_xy": [float(x), 20.0], "tracking_state": "measured"},
    }


def event(index, hitter, sequence_support=1.0):
    return {
        "event_id": f"evt_{index}",
        "candidate_frame": index,
        "predicted_hitter": hitter,
        "play_state_sequence_support": sequence_support,
    }


class PlayStateGateTests(unittest.TestCase):
    def test_keeps_alternating_candidates_connected_by_flight(self):
        records = [frame(index, index * 2) for index in range(31)]
        kept, suppressed, summary = apply_play_state_gate(
            records, [event(5, "upper"), event(20, "lower")], fps=10,
        )
        self.assertEqual([row["candidate_frame"] for row in kept], [5, 20])
        self.assertEqual(suppressed, [])
        self.assertEqual(summary["kept_count"], 2)
        self.assertEqual(records[10]["play_state"], "active_play")

    def test_suppresses_singleton_waiting_candidate_but_preserves_audit(self):
        records = [frame(index, index) for index in range(20)]
        kept, suppressed, summary = apply_play_state_gate(records, [event(8, "upper")], fps=10)
        self.assertEqual(kept, [])
        self.assertEqual([row["candidate_frame"] for row in suppressed], [8])
        self.assertEqual(suppressed[0]["play_state_gate_decision"], "suppressed_inactive_wait")
        self.assertEqual(summary["suppressed_count"], 1)

    def test_same_hitter_candidates_do_not_form_active_rally(self):
        records = [frame(index, index * 2) for index in range(31)]
        kept, suppressed, _ = apply_play_state_gate(
            records, [event(5, "upper"), event(20, "upper")], fps=10,
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(suppressed), 2)

    def test_isolated_low_support_candidate_cannot_open_active_segment(self):
        records = [frame(index, index * 2) for index in range(41)]
        kept, suppressed, summary = apply_play_state_gate(
            records,
            [event(2, "upper", 0.5), event(15, "upper"), event(30, "lower")],
            fps=10,
        )
        self.assertEqual([row["candidate_frame"] for row in kept], [15, 30])
        self.assertEqual([row["candidate_frame"] for row in suppressed], [2])
        self.assertEqual(summary["segments"][0]["start_frame"], 13)

    def test_disabled_gate_is_lossless(self):
        records = [frame(index, index) for index in range(10)]
        source = [event(5, "upper")]
        kept, suppressed, summary = apply_play_state_gate(records, source, fps=10, enabled=False)
        self.assertEqual(kept, source)
        self.assertEqual(suppressed, [])
        self.assertFalse(summary["enabled"])

    def test_rejects_invalid_link_window(self):
        with self.assertRaisesRegex(ValueError, "min < max"):
            apply_play_state_gate([], [], fps=30, min_link_seconds=4, max_link_seconds=4)


if __name__ == "__main__":
    unittest.main()
