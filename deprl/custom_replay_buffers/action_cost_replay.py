import numpy as np

from deprl.vendor.tonic import logger
from deprl.vendor.tonic.replays import Buffer


class AdaptiveEnergyBuffer(Buffer):
    """
    Assume all activity is appended at the end of the observation.
    True for Myosuite and scone so far.
    """

    def __init__(self, *args, **kwargs):
        # User parameters - num_acts needs to match your model!
        self.num_acts = kwargs.pop("num_acts")
        # smoothing parameter
        self.alpha = kwargs.pop("alpha", 0.8)
        # performance threshold that needs to be achieved
        self.threshold = kwargs.pop("threshold", 1000)
        # initial learning rate for the energy cost
        self.lr = kwargs.pop("lr", 9e-4)
        # how much the learning rate decreased after a collapse
        self.lr_decimation = kwargs.pop("lr_decimation", 0.9)
        # type of energy cost fn
        self.cost_function = kwargs.pop("cost_function", 3)

        # Initial values 
        self.action_cost = 0.0
        self.cdt_avg = 0
        self.score_avg = 0
        super().__init__(*args, **kwargs)
        self.checkpoint_fields = [
            "index",
            "num_workers",
            "max_size",
            "size",
            "buffers",
            "action_cost",
            "lr",
            "cdt_avg",
            "score_avg",
        ]

    def store(self, **kwargs):
        if "env_infos" in kwargs.keys():
            # remove it for tonic compliance
            kwargs.pop("env_infos")
        super().store(**kwargs)

    def get(self, *keys, steps):
        """Get batches from named buffers."""
        for _ in range(self.batch_iterations):
            total_size = self.size * self.num_workers
            indices = np.random.randint(total_size, size=self.batch_size)
            rows = indices // self.num_workers
            columns = indices % self.num_workers
            batch = {k: self.buffers[k][rows, columns] for k in keys}
            yield self._relabel_batch(batch, rows, columns)
        self.last_steps = steps


# From the Natural and Robust Walking paper. Called from custom_trainer.py
# on every worker 0 train episode end, not eval. Tracks a smoothed average
# of worker 0's raw score against threshold; ramps action_cost up once
# consistently above it (penalize energy now that walking works), back
# down otherwise.
    def adjust(self, score):
        self.score_avg = self.alpha * self.score_avg + (1 - self.alpha) * score
        if self.score_avg > self.threshold and self.cdt_avg < 0.5:
            self.lr = self.lr * self.lr_decimation
            self.action_cost += self.lr
        elif self.score_avg > self.threshold and self.cdt_avg > 0.5:
            self.action_cost += self.lr
        elif self.score_avg < self.threshold:
            self.action_cost -= self.lr
        else:
            raise Exception
        delta_cdt = [1 if self.score_avg > self.threshold else 0][0]
        self.cdt_avg = self.alpha * self.cdt_avg + (1 - self.alpha) * delta_cdt
        self.action_cost = np.clip(self.action_cost, 0, 100)
        logger.store("train/energy_buffer/self.score_avg", self.score_avg)
        logger.store("train/energy_buffer/lr", self.lr)
        logger.store("train/energy_buffer/prev_cdt", self.cdt_avg)
        logger.store(
            "train/energy_buffer/action_cost_intern", self.action_cost
        )

#every sampled batch gets reward -= action_cost*cost.
#FYI: this happens only at sampling time and not when the transition is stored. 
#Same stored transition could be relabeled differently bc action_cost changes over training.
    def _relabel_batch(self, batch, rows, columns):
        cost = self.action_cost * self._get_cost(batch["next_observations"])
        batch["rewards"] = batch["rewards"] - cost
        logger.store(
            "train/energy_buffer/avg_relabel_action_cost", np.mean(cost)
        )
        return batch


#reads activations from the tail of the observation vector and calculates energy cost
    def _get_cost(self, observations):
        """
        Muscle activity is assumed to be the last observation in the state!
        """
        acts = observations[:, -self.num_acts :]  #tail
        if self.cost_function == 0:
            return 4.0 * np.sum([np.power(acts, k) for k in range(1, 5)])
        if self.cost_function == 1:
            return 4.0 * np.mean([np.power(acts, k) for k in range(1, 5)])
        if self.cost_function == 2:
            return np.mean(np.power(acts, 3))
        if self.cost_function == 3:  #default
            return np.sum(np.power(acts, 3))
        if self.cost_function == 4:
            return -np.sum(np.exp(-5 * acts))