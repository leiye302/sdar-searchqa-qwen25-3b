"""Compare the official launchers and the already-published beta5 recipe."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import UPSTREAM, load
from run import config


def launcher(path):
    """Resolve only literal shell assignments; never execute upstream shell."""
    source = (UPSTREAM / path).read_text(encoding='utf-8')
    assignments = dict(re.findall(r'^(\w+)=([^\n]+)$', source, re.M))
    assignments['ENGINE'] = 'vllm'
    tokens = shlex.split(source.split('python3 -m ', 1)[1].replace('\\\n', ' '))
    entry = tokens.pop(0)
    assert entry == 'verl.trainer.main_sdar'
    options = {}
    for token in tokens:
        if token == '$@':
            continue
        key, value = token.lstrip('+').split('=', 1)
        if re.fullmatch(r'\$\w+', value):
            value = shlex.split(assignments[value[1:]])[0]
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        else:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        assert key not in options
        options[key] = value
    return options


def test_pinned_official_launchers_differ_only_in_gate_and_label():
    lock = load('native_comparison.json')
    sdar = launcher(lock['native_sdar_launcher'])
    opsd = launcher(lock['native_grpo_opsd_launcher'])
    assert sdar.keys() == opsd.keys()
    delta = {k: {'sdar': sdar[k], 'grpo_opsd': opsd[k]} for k in sdar
             if sdar[k] != opsd[k] and k != 'trainer.experiment_name'}
    assert delta == lock['official_scientific_difference'] == {
        'algorithm.sdar.gate_beta': {'sdar': 5., 'grpo_opsd': 0.}}


def test_every_native_scientific_override_is_retained(tmp_path):
    from omegaconf import OmegaConf
    size = load('release.json')['size']
    model, run = tmp_path / 'base', tmp_path / 'run'
    cfg = config(size, model, run)
    native = launcher(load('upstream.json')['native_launcher'])
    identity_and_evaluation = {
        'data.train_files', 'data.val_files', 'actor_rollout_ref.model.path',
        'algorithm.sdar.skills_dir', 'env.search.search_url', 'trainer.logger',
        'trainer.project_name', 'trainer.experiment_name', 'trainer.save_freq',
        'trainer.test_freq', 'trainer.val_before_train',
    }
    assert identity_and_evaluation <= native.keys()
    for key in sorted(native.keys() - identity_and_evaluation):
        assert OmegaConf.select(cfg, key) == native[key], key
    assert cfg.actor_rollout_ref.model.path == str(model)
    assert cfg.algorithm.sdar.skills_dir == str(UPSTREAM / 'skills/search')
    assert cfg.trainer.test_freq == cfg.trainer.save_freq == 20
    assert cfg.trainer.val_before_train is True
    assert cfg.data.val_files.endswith('validation_subset.parquet')


def test_equivalence_to_published_beta5_is_explicit_and_verified():
    lock = load('native_comparison.json')
    assert lock['same_treatment_as_published_beta5'] is True
    excluded = set(lock['identity_only_experiment_keys'])
    assert excluded == {'trainer.project_name', 'trainer.experiment_name'}
    science = {k: v for k, v in load('experiment.json').items() if k not in excluded}
    identity = hashlib.sha256(json.dumps(science, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert identity == lock['published_beta5_recipe_sha256']
