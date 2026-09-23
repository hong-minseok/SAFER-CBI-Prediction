# -*- coding: utf-8 -*-
"""Shared HPO helpers for suggestions and baseline extraction."""

from typing import Any, Dict, Optional

import optuna


def _suggest_from_space(trial: optuna.Trial, space: Dict[str, Any],
                        conditional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Suggest hyperparameters from a JSON-defined search space."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        t = spec.get('type')
        if t == 'categorical':
            params[name] = trial.suggest_categorical(name, spec['choices'])
        elif t == 'int':
            params[name] = trial.suggest_int(name, int(spec['low']), int(spec['high']))
        elif t == 'float':
            if spec.get('log', False):
                params[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']), log=True)
            else:
                params[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']))
    if conditional:
        for name, spec in conditional.items():
            choices = list(spec['choices'])
            dep = spec.get('depends_on')
            if dep and spec.get('filter') == 'divisible' and dep in params:
                filtered = [c for c in choices if params[dep] % c == 0]
                choices = filtered or choices[:1]
            params[name] = trial.suggest_categorical(name, choices)
    return params


def _extract_baseline_params(parameters: Dict[str, Any],
                              conditional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract midpoint parameters from search space for baseline trial."""
    baseline: Dict[str, Any] = {}
    for name, spec in parameters.items():
        t = spec.get('type')
        if t == 'categorical':
            choices = spec['choices']
            baseline[name] = choices[len(choices) // 2]
        elif t == 'int':
            baseline[name] = (int(spec['low']) + int(spec['high'])) // 2
        elif t == 'float':
            low, high = float(spec['low']), float(spec['high'])
            baseline[name] = (low + high) / 2
    if conditional:
        for name, spec in conditional.items():
            choices = spec['choices']
            dep = spec.get('depends_on')
            if dep and spec.get('filter') == 'divisible' and dep in baseline:
                choices = [c for c in choices if baseline[dep] % c == 0] or choices[:1]
            baseline[name] = choices[len(choices) // 2]
    return baseline
