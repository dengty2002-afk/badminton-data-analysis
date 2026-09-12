# Local input and generated data

Register your own video with `python -m badminton_pipeline.ingest.register_video --input <video> --data-dir data`.

Registrations contain local file paths. Video registrations, calibrations, raw/derived frames, full research batches and Gold inputs remain local and are ignored by Git. Synthetic examples live in `examples/`; published aggregate metrics live with their experiment or research report.
