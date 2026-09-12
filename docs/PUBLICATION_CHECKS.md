# Publication checks — 2026-09-13

The publication snapshot was built separately from the working project. The original 200 copied source/configuration/report files were hash-checked and remained unchanged.

Completed locally:

- Installed the package with `pip install -e .` into a new, isolated Python 3.12 environment containing only the declared base dependencies.
- Ran the synthetic event-matching example: TP=1, FP=1, FN=1, Precision=Recall=F1=0.5, as expected.
- Initially ran 82 Python tests: 81 passed, 1 explicitly skipped because separately downloaded model weights were absent. A remote Windows run then exposed an absolute-path alias inconsistency in batch resume. The publication copy now canonicalizes accepted absolute calibration paths before comparison; a regression test covers the alias case. The updated local clean-environment suite runs **83 tests: 82 passed, 1 explicitly skipped**.
- Parsed the Windows setup script successfully. The full GPU dependency download/installation was not repeated in this publication task.
- Verified the four optional SwingNet source downloads against their pinned SHA256 values. Those downloaded files are ignored by Git and are not bundled.
- Checked publication candidates for oversized files, known credential patterns, private-key headers and machine-specific home paths. No matching credentials were found in the files being published. This pattern check is not a comprehensive security audit.
- Preserved the existing repository's commit history and older plan documents. The owner explicitly approved publishing that previously private history. Its historical local agent configuration contains a machine-home path; the current version removes that local configuration and ignores future `.claude/` files. History has not been rewritten.

Not claimed:

- No new BST accuracy evaluation, video benchmark, GPU inference or 24-match research rerun.
- No full frontend runtime/build validation or website deployment.
- No cross-platform guarantee. The first Linux Actions run exposed the batch runner's existing dependency on Windows `msvcrt`. CI now runs the same complete test suite on `windows-latest`, matching the supported platform; Linux batch execution has not been ported.

The source checkout excludes original match videos, model binaries, raw frame data, virtual environments, caches, and local input manifests. Published numerical experiment results retain their previously documented retrospective and measurement-error limitations.
