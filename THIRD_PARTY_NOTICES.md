# Third-party notices

Safe Stage Python tools and documentation are licensed under [LICENSE](LICENSE).
Third-party objects, textures, materials, fixtures, robot geometry, backgrounds,
and derived media retain their respective terms.

The asset manifest [`catalog/assets.json`](catalog/assets.json) identifies the
required upstream resources, dependency paths, and license texts. Available
license and attribution documents are retained verbatim in
[`licenses/third_party/`](licenses/third_party/):

- RoboLab
- YCB assets
- HOT3D assets
- HOPE assets
- HANDAL assets

These terms differ across asset sets. The Safe Stage software license does not
replace them or grant a blanket right to redistribute the external bundle.
Assets without sufficiently specific license notices remain marked
`requires_source_license_review` in the manifest.

The optional `data/objects`, `data/robots`, and `data/backgrounds` directories
are excluded from Git, source packages, and wheels. See
[asset setup](docs/ASSETS.md) for obtaining and verifying a local bundle.
Asset references use relative paths for portability. Mesh geometry and physics
parameters retain their upstream values.

Separately installed dependencies, including Isaac Sim, Isaac Lab, RoboLab,
OpenUSD, and NumPy, retain their own licenses.
