"""Hydra utils."""

from hydra.core.utils import setup_globals
from hydra.utils import get_class
from omegaconf import OmegaConf


def define_resolvers():
    # Resolvers used in hydra configs (see
    # https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
    if not OmegaConf.has_resolver("eval"):
        # pylint: disable = import-outside-toplevel, eval-used
        import numpy as np
        OmegaConf.register_new_resolver(
            "eval",
            # Makes numpy available for use in eval.
            lambda source: eval(source, {"np": np}),
        )

    # Allows us to resolve default arguments which are copied in multiple places
    # in the config.
    if not OmegaConf.has_resolver("resolve_default"):
        OmegaConf.register_new_resolver(
            "resolve_default", lambda default, arg: default
            if arg == "" else arg)

    # For converting things like numpy.float32 to actual classes.
    if not OmegaConf.has_resolver("get_cls"):
        OmegaConf.register_new_resolver(name="get_cls", resolver=get_class)

    # Hydra's base resolvers (these seem to not be loaded when experimental
    # reload is used).
    if not OmegaConf.has_resolver("hydra"):
        setup_globals()
