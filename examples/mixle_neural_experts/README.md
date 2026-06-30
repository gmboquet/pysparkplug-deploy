# Train a mixle model with a neural component

A runnable example of the **`--backend mixle`** training path: the artifact is a *mixle model* (a
`MixtureDistribution`) whose experts are **neural nets** (`NeuralLeaf` = a torch module wrapped as a
generative `p(y|x)` leaf). mixle fits the whole thing with EM — E-step computes responsibilities, M-step
trains each expert by responsibility-weighted gradient descent. The neural net is **one part** (a leaf)
of the mixle model.

On a two-regime synthetic problem (`y = +2x` vs `y = -2x`), the two experts must specialize — a single
network can't fit both. Training prints the learned per-expert slopes (≈ `-2` and `+2`).

## Run it

Locally (no GPU rental, no spend):
```sh
mixle-mlops train neural-experts --local --backend mixle \
    --script train.py --workdir examples/mixle_neural_experts
```
On a rented GPU (same command, drop `--local`, set `MIXLE_VAST_API_KEY`). The box installs the mixle
core from git by default (`git+https://github.com/gmboquet/mixle.git@evolve`) so it runs the current
code, not a pinned PyPI release — override with `--mixle-git` or the `MIXLE_GIT` env var:
```sh
MIXLE_VAST_API_KEY=... mixle-mlops train neural-experts --backend mixle \
    --script train.py --workdir examples/mixle_neural_experts \
    --gpu RTX_4090 --max-price 0.5 --max-runtime 20 --no-dry-run
```

The script writes the trained model to `--output` and reloads it to confirm. **Note:** as written the
`NeuralLeaf` experts train on CPU; to actually exercise the GPU the leaf modules need to be moved to CUDA.

## Making the neural part an actual LLM

Swap the MLP `NeuralLeaf` for one of mixle's sequence leaves to make the component a real language model:
`mixle.models.transformer` / `streaming_transformer_leaf` (a transformer as a leaf) or `dpo_leaf`
(preference-tuned LM as a leaf). The multi-stage path — fine-tune an LLM first (see the LLM backend),
then use it as the leaf — slots in at the model-construction step here.

## Persistence note

mixle's JSON model registry can't serialize torch-backed leaves (arbitrary `nn.Module`s aren't
JSON-reconstructable), so neural mixle models are saved with `pickle` + a JSON provenance summary taken
from the `fit_with_provenance` header. (Pure-stats mixle models can use the JSON `Registry` directly.)
