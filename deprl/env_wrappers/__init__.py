import deprl  # noqa
from deprl.env_wrappers.dm_wrapper import DMWrapper, OstrichDMWrapper
from deprl.env_wrappers.gym_wrapper import GymWrapper
from deprl.env_wrappers.scone_wrapper import SconeWrapper

try:
    # Optional -- only needed so eval(env) below can resolve
    # leaps.envs.LatentActionPriorWrapper for configs that use it. Soft
    # dependency: this file is shared by every environment deprl touches,
    # not just LEAPS ones, so don't hard-fail if leaps isn't installed.
    import leaps.envs  # noqa
except ImportError:
    pass


def apply_wrapper(env):
    if "control" in str(env).lower():
        if env.name == "ostrich-run":
            return OstrichDMWrapper(env)
        return DMWrapper(env)
    elif "scone" in str(env).lower():
        return SconeWrapper(env)
    else:
        return GymWrapper(env)


def env_tonic_compat(env, id=5, parallel=1, sequential=1):
    """
    Applies wrapper for tonic and passes random seed.
    """
    return apply_wrapper(eval(env))


__all__ = [env_tonic_compat, apply_wrapper]
