# Third-party provenance

This repository does not replace the licenses of its dependencies, source extracts, models, or datasets.

| Component | Pinned source | Treatment |
|---|---|---|
| Good-Badminton court code | `yo-WASSUP/Good-Badminton`, commit `24f2208929f84f441d7af3bdb1ba6bf6c109943f` | Minimal court runtime included; Apache-2.0 text retained in `vendor/Good-Badminton/LICENSE` |
| BST model runtime | `Va6lue/BST-Badminton-Stroke-type-Transformer`, commit `fb9b310bf4c8a8e3d89c75e61bc06a7ac3de62df` | Minimal model code included; MIT text retained in its vendor directory |
| MonoTrack experimental sources | `jhwang7628/monotrack`, commit `39e9227bd44fa2a579731b1cc97c7f529f391cd6` | Adobe Research License retained in `experiments/monotrack-hitnet/upstream/LICENSE`; noncommercial research limitations apply to those materials and derivatives |
| GolfDB / badminton SwingNet preprocessing | Fixed URLs and content SHA256 in `experiments/swingnet/upstream/sources.json` | No verified redistribution license was established for the required source snapshot; original files are not bundled. `fetch_upstream.py` retrieves them from their authors for local use |
| SwingNet checkpoint | [Zenodo record 14677727](https://zenodo.org/records/14677727) | Binary not bundled; extraction script and checkpoint SHA256 documented |
| BST / Good-Badminton weights | `configs/models.bst.json`, `configs/models.good-badminton.json` | Binaries not bundled; download URLs and hashes retained. Code licenses do not automatically establish weight or training-data permissions |

Upstream links: [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton), [BST](https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer), [MonoTrack](https://github.com/jhwang7628/monotrack), [GolfDB](https://github.com/wmcnally/golfdb), [badminton SwingNet](https://github.com/wish44165/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection).

Other runtime dependencies retain their own package licenses. In particular, importing a package does not relicense it under the project's terms. Consult the pinned dependency distributions for their current notices.
