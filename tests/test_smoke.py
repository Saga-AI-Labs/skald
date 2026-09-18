"""Smoke tests — verify package layout imports and interface signatures."""

from __future__ import annotations

import inspect

import adapters
import identity
import store


def test_adapter_protocol_has_run():
    assert hasattr(adapters.SuiteAdapter, "run")
    sig = inspect.signature(adapters.SuiteAdapter.run)
    params = list(sig.parameters)
    # self, model, task, config
    assert "model" in params
    assert "task" in params
    assert "config" in params


def test_store_put_callable():
    assert callable(store.put)


def test_store_query_callable():
    assert callable(store.query)


def test_identity_hash_checkpoint_callable():
    assert callable(identity.hash_checkpoint)
