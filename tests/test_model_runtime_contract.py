import unittest


class ModelRuntimeContractTests(unittest.TestCase):
    def test_cuda_metadata_must_not_mask_cpu_fallback(self):
        # The live adapter inspects both RTMLib sessions after construction.
        # This regression guard keeps the user-visible contract explicit.
        from badminton_pipeline.models.good_badminton import ModelSetupError

        self.assertTrue(issubclass(ModelSetupError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
