# Autoresearch loop

This package runs generation-based phase2 and phase3 optimization campaigns from
the repository root. Each trial gets an isolated output directory under
`autoresearch/output/<campaign_id>/generation_<NN>/run_<run_id>/`.

Common commands:

```bash
python -m autoresearch.plan --config autoresearch/configs/default.yaml --dry-run
python -m autoresearch.run_loop --config autoresearch/configs/default.yaml --gpus 0,1,2,3
python -m autoresearch.render_dashboard --state-dir autoresearch/output/latest
```

The default config is JSON-compatible YAML because this environment does not
ship PyYAML. It can still be edited as a normal structured config, and the
loader uses only the Python standard library.

The loop is restartable. On startup, records left in `running` state are marked
failed, completed runs are skipped, and pending trials are launched or resumed.
Progress is tracked in:

- `runs.jsonl`: append-only run log.
- `state.json`: compact latest state and rank summary.
- `index.html`: static dashboard with metrics, config, artifacts, and logs.

Each subprocess receives a single GPU through `CUDA_VISIBLE_DEVICES`. The local
scheduler starts at most one process per configured GPU id.
