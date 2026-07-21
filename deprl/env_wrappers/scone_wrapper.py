import gym
import numpy as np

from .wrappers import ExceptionWrapper


class CustomSconeException(Exception):
    """
    Custom exception class for Scone. The SconePy Interface doesn't define a scone exception
    at the moment and I have never encountered a simulator-based exception until now.
    Will update when that is the case.
    """

    pass


class SconeWrapper(ExceptionWrapper):
    """Wrapper for SconeRL, compatible with
    gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error = CustomSconeException

    def render(self, *args, **kwargs):
        pass

    def muscle_lengths(self):
        length = self.unwrapped.model.muscle_fiber_length_array()
        return length

    def muscle_forces(self):
        force = self.unwrapped.model.muscle_force_array()
        return force

    def muscle_velocities(self):
        velocity = self.unwrapped.model.muscle_fiber_velocity_array()
        return velocity

    def muscle_activity(self):
        return self.unwrapped.model.muscle_activation_array()

    def write_now(self):
        if self.unwrapped.store_next:
            self.model.write_results(
                self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
            )
        self.episode += 1
        self.unwrapped.store_next = False

    def _muscle_param_diagnostics(self):
        """Physical-parameter compensation tracking, computed unconditionally
        for every SconeWrapper (baseline runs included, not just latent-prior
        ones), so the two are directly comparable.

        Uses Muscle.force_norm()/fiber_length_norm() directly (SCONE's own
        normalized accessors) rather than hand-dividing force_array() by
        max_isometric_force() -- a first version did that and produced
        physically implausible values (fiber length norm ~11, force
        utilization ~1e-5), almost certainly because SCONE's internal
        normalization accounts for things (pennation angle etc.) a naive
        ratio doesn't. max_isometric_force/optimal_fiber_length are still
        real per-muscle constants of the model, just not what we divide by
        here.

        No moment-arm / torque-contribution metric here -- this sconepy
        build exposes neither Muscle.dofs() nor any moment-arm accessor on
        Muscle or Model (checked via dir() on real objects, not assumed).

        Not cached across steps: 18 muscles is cheap next to the physics
        integration step, and caching risks stale muscle handles surviving a
        reset() -- simplicity over a micro-optimization we have no evidence
        we need.
        """
        model = self.unwrapped.model
        muscles = model.muscles()

        force_norm = np.array([m.force_norm() for m in muscles])
        fiber_length_norm = np.array([m.fiber_length_norm() for m in muscles])

        return {
            "train/muscle_force_utilization_mean": float(force_norm.mean()),
            "train/muscle_force_utilization_max": float(force_norm.max()),
            "train/muscle_fiber_length_norm_mean": float(fiber_length_norm.mean()),
            "train/muscle_fiber_length_norm_dev_from_optimal": float(np.mean(np.abs(fiber_length_norm - 1.0))),
        }

    def _inner_step(self, action):
        """
        takes an action and advances environment by 1 step.
        Changed to allow for correct sto saving.
        """
        if not self.unwrapped.has_reset:
            raise Exception("You have to call reset() once before step()")

        # self.unwrapped.model below skips every wrapper in between, so any
        # ActionWrapper stacked under us (e.g. LatentActionPriorWrapper) would
        # otherwise never run. Walk down and apply their .action() transforms
        # first, same order they'd compose in if the chain wasn't bypassed.
        # No-op when there's no ActionWrapper in the chain (e.g. baseline
        # configs with no wrapper between SconeWrapper and the base env).
        env = self.env
        step_info = {}
        while hasattr(env, "env"):
            if isinstance(env, gym.ActionWrapper):
                action = env.action(action)
            env = env.env

        # HERE! clip_hi pulled out to a variable (was inlined as a literal
        # 0.5/1.0 in the clip call below) so it can be reused for the
        # post-clip diagnostics pass right after -- same clip bound, not a
        # second guess at it.
        clip_hi = 0.5 if self.unwrapped.clip_actions else 1.0
        action = np.clip(action, 0, clip_hi)

        # HERE! Diagnostics collection moved to *after* the clip above,
        # instead of being read off `_last_diagnostics` inside the
        # pre-clip .action() loop that used to be here. Any ActionWrapper can
        # expose a `_diagnostics_post_clip(clip_lo, clip_hi)` method to
        # compute its own metrics against the clip bound actually applied to
        # the muscles -- see
        # leaps/envs/latent_env.py::LatentActionPriorWrapper for why this
        # matters (pre-clip diagnostics were measuring values the simulator
        # never saw).
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "_diagnostics_post_clip"):
                step_info.update(env._diagnostics_post_clip(0, clip_hi))
            env = env.env

        if self.unwrapped.use_delayed_actuators:
            self.unwrapped.model.set_delayed_actuator_inputs(action)
        else:
            self.unwrapped.model.set_actuator_inputs(action)

        self.unwrapped.model.advance_simulation_to(self.time + self.step_size)
        reward = self.unwrapped._get_rew()
        obs = self.unwrapped._get_obs()
        done = self.unwrapped._get_done()
        self.unwrapped.time += self.step_size
        self.unwrapped.total_reward += reward

        # Unconditional (unlike the ActionWrapper diagnostics above) --
        # applies to baseline runs too, using only self.unwrapped.model, so
        # baseline vs. latent-prior are directly comparable on these.
        step_info.update(self._muscle_param_diagnostics())
        # NOTE: Our wrapper uses gym style "done" 
        # but if we migrate to gymnasium we should update method 
        # to return (obs, reward, terminated, truncated, info)
        # also make sure `ExceptionWrapper.step()` computes `done= terminated or truncated`.
        return obs, reward, done, step_info

    @property
    def _max_episode_steps(self):
        return 1000

    @property
    def results_dir(self):
        return self.unwrapped.results_dir
