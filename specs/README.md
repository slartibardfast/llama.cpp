# Behavioural (.allium) specs for the llama.cpp autoround fork

These specs constrain behaviour of this fork of `llama.cpp` (currently the
`GGML_TYPE_Q4_0_AR16` quant type ported from ik_llama.cpp). They live beside the
software per the host methodology (template ledger entry `b6232a5`), on the
`host-specs` branch. Unlike ik_llama.cpp's orphan `host-specs` branch, this one is
cut from the `autoround` code line, so the specs, the discharging binding fixtures,
and the code under test sit together and the obligations resolve for real (the
merge that `call/0013` names as the mechanism for substantive discharge).

## Layout

```
specs/<topic>/           one dir per concern (mmq, ...)
  <name>.allium          behavioural spec (allium 3)
  <name>.obligations     per-obligation disposition manifest (allium plan)
  <name>_binding.c       discharging fixture, built against this tree's ggml
  run-binding.sh         build + run the fixture (CPU-only static build)
```

## Lanes

- **allium** (requirements): `allium check` + `allium analyse` are green on every
  spec. `allium plan` derives obligations; each is dispositioned in a sibling
  `.obligations` manifest — `test:<name>` where the binding fixture discharges it
  against the landed C reference, `waived:` with the concrete reason where nothing
  dischargeable exists yet (FP16-accumulator bounds pending the CUDA port,
  C-side precondition contracts).
- **binding fixture** (discharge): `specs/mmq/run-binding.sh` builds ggml CPU-only
  (static, ccache off) and runs the property checks bit-exact against the scalar
  reference. The quantize tie semantics were reconciled spec-to-C (the ik spec
  text said round-nearest-even; both forks' C rounds ties away from zero on a
  reciprocal multiply, and the C is the accepted ground truth) — the amended
  QuantizeFormula records the ruling, and `check_quantize_tie_semantics` pins
  the tie behaviour so a silent move to RNE would fail the lane.

CI: `.github/workflows/spec-gates.yml` (push to `host-specs`). The host references
this branch by pin in `.host-software`.
