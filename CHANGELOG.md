# Changelog

## [v0.1.0] - 2026-10-31

First curated code release of TextFlow, associated with *Towards Training-Free Scene Text Editing*.

### Included

- FLUX-Kontext implementations and a LongCat adaptation, with task/role-based entry and module names and bilingual usage documentation.
- Sanitized local paths and scoped runtime fixes for CLI validation, stable sample identities/seeds, configuration-guarded resume, atomic image output and mocked multi-rank control flow; method stages and weights are unchanged.
- 27 lightweight CPU regression tests: 11 entry/helper checks, 8 FLUX pipeline checks and 8 LongCat pipeline checks. Pipeline checks use AST extraction and stubs rather than model inference.
- Ubuntu CI for Python 3.10 and 3.11, installing only Pillow==11.3.0 as a test dependency; no torch installation or model download.

### Known boundaries

- This is a curated research snapshot, not an architectural refactor or a claim of paper-result reproduction.
- Dependency installation for inference, GPU inference, actual distributed/NCCL execution, image quality, performance and memory requirements remain unvalidated.
- No model weights, datasets, benchmark evaluation suite or vendored diffusers checkout are included. Model and dataset terms apply separately.
- LongCat FlowEdit support remains experimental. Stable sample seeds do not guarantee bitwise reproducibility across hardware or dependency versions.
