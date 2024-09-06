from typing import List, Dict, Callable

import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list, treedef_list = list(zip(*[tree_flatten(tree) for tree in trees]))
    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


class RngKeyMixin:

    def __init__(self, rng_key: jax.random.PRNGKey):
        self._rng_key = rng_key

    def _next_rng_key(self) -> jax.random.PRNGKey:
        new_key, self._rng_key = jax.random.split(self._rng_key)
        return new_key

    @property
    def rng_key(self) -> jax.random.PRNGKey:
        return self._next_rng_key()


""" Stats aggregation """

def aggregate_stats(stats_list: List[Dict]) -> Dict:
    return jax.tree_util.tree_map(jnp.mean, tree_stack(stats_list))

""" Root finding """

def find_root_1d(fun: Callable, low: float = -1e6, high: float = 1e6,
              atol: float = 1e-6, maxiter: int = 10**2):
    assert fun(low) < 0, 'Lower bound not small enough'
    assert fun(high) > 0, 'High bound not small enough'

    for i in range(maxiter):
        middle = (high + low) / 2.0
        if jnp.abs(low - high) < atol:
            return middle
        f_middle = fun(middle)
        if f_middle > 0:
            high = middle
        else:
            low = middle

    raise RuntimeError(f'Reached max iterations of {maxiter} without reaching the atol of {atol}.')