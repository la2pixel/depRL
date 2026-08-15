import gym
import numpy as np

from .wrappers import ExceptionWrapper


class CustomSconeException(Exception):
    """
    Custom exception class for Scone. The SconePy Interface doesn't define a scone exception atm
    """

    pass


class SconeWrapper(ExceptionWrapper):
    """Wrapper for SconeRL, compatible with gym=0.13.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sconepy has no exception type of its own; this is a placeholder
        # for ExceptionWrapper.step()'s except clause (see _inner_step's
        # NaN check below).
        self.error = CustomSconeException

    def render(self, *args, **kwargs):
        pass

    # Satisfy AbstractWrapper's muscle_lengths()/muscle_forces() abstract
    # methods. DEP's exploration controller reads these when mix > 0.

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
        """Writes a .sto file for the current episode if store_next_episode()
        was called. _inner_step() bypasses GaitGym.step() entirely, so this
        stands in for the write-on-done branch the base env would run."""
        if self.unwrapped.store_next:
            self.model.write_results(
                self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
            )
        self.episode += 1
        self.unwrapped.store_next = False

    def _muscle_param_diagnostics(self):
        """Compensation tracking, same for baseline and LAP runs.

        Uses force_norm()/fiber_length_norm() directly. Dividing
        force_array() by max_isometric_force() gave implausible numbers
        (fiber length norm ~11); SCONE's own normalization must account
        for pennation angle etc.
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
        """Advances the env by one step. Reimplements GaitGym.step() rather
        than calling it, since the clip below has to run after any
        ActionWrapper transform (see the walk right below).

        Skips _apply_termination_cost (a no-op in every current sconegym
        subclass) and the done-branch .sto write, which write_now() above
        replaces, called only from play.py/examples, never from training.
        """
        if not self.unwrapped.has_reset:
            raise Exception("You have to call reset() once before step()")

        # self.unwrapped.model skips every wrapper below us, so any
        # ActionWrapper (e.g. LatentActionPriorWrapper) needs its .action()
        # applied manually here. No-op if there's no ActionWrapper in the
        # chain, e.g. baseline configs.
        env = self.env
        step_info = {}
        while hasattr(env, "env"):
            if isinstance(env, gym.ActionWrapper):
                action = env.action(action)
            env = env.env

        # clip_hi pulled out so the post-clip diagnostics below reuse the
        # same bound instead of guessing at it again.
        clip_hi = 0.5 if self.unwrapped.clip_actions else 1.0
        action = np.clip(action, 0, clip_hi)

        # Diagnostics read after the clip now, not before, so they reflect
        # what the simulator actually saw. See
        # leaps/envs/latent_env.py::LatentActionPriorWrapper for the
        # _diagnostics_post_clip(clip_lo, clip_hi) hook.
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "_diagnostics_post_clip"):
                step_info.update(env._diagnostics_post_clip(0, clip_hi))
            env = env.env

        # Hand the final, clipped action to sconepy directly, mirrors
        # GaitGym.step()'s own actuator-setting branch. use_delayed_actuators
        # models actuation delay, off in every config in this project.
        if self.unwrapped.use_delayed_actuators:
            self.unwrapped.model.set_delayed_actuator_inputs(action)
        else:
            self.unwrapped.model.set_actuator_inputs(action)

        # Same order as GaitGym.step(): advance physics one step_size, then
        # read reward/obs/done off the post-step model state.
        self.unwrapped.model.advance_simulation_to(self.time + self.step_size)
        reward = self.unwrapped._get_rew()
        obs = self.unwrapped._get_obs()
        done = self.unwrapped._get_done()
        self.unwrapped.time += self.step_size
        self.unwrapped.total_reward += reward

        # Unconditional, unlike the ActionWrapper diagnostics above; uses
        # only self.unwrapped.model, so baseline runs get these too.
        step_info.update(self._muscle_param_diagnostics())
        # NOTE: Our wrapper uses gym style "done"
        # but if we migrate to gymnasium we should update method
        # to return (obs, reward, terminated, truncated, info)
        # also make sure `ExceptionWrapper.step()` computes `done= terminated or truncated`.
        return obs, reward, done, step_info

    @property
    def _max_episode_steps(self):
        # Hardcoded: every model here (h0918/h1622/h2190) uses
        # step_size=0.025s, so 1000 steps is 25s regardless of the model.
        return 1000

    @property
    def results_dir(self):
        # AbstractWrapper defaults this to None; delegate to the base env
        # so SCONE runs actually get a results_dir.
        return self.unwrapped.results_dir
