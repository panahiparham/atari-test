import numpy as np
import pickle
import os

from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from environments.Atari import Atari as AtariEnv
from PyExpUtils.collection.Collector import Collector
from ReplayTables.storage.CompressedStorage import CompressedStorage

def upperFirst(s: str):
    f = s[0].upper()
    return f + s[1:]

def toGymStr(s: str):
    if '_' in s:
        parts = s.split('_')
        return ''.join(map(upperFirst, parts))

    return upperFirst(s)

class Atari(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        game = toGymStr(self.env_params['game'])

        self.env = AtariEnv(game, self.seed)
        self.actions = self.env.num_actions()

        self.observations = (84, 84, 4)
        self.gamma = 0.99
        self.params['observation_type'] = np.uint8

        # enable reward clipping
        self.params['reward_clip'] = 1

    def getAgent(self):
        agent = super().getAgent()

        # set-up compressed storage for experience replay
        if hasattr(agent, 'buffer'):
            storage = CompressedStorage(agent.buffer._max_size)
            agent.buffer.use_storage(storage)

        return agent

    def maybe_save_agent(self, step: int):
        save_interval = self.exp.save_interval
        if save_interval <= 0:
            return

        if step % save_interval != 0:
            return

        save_path = os.path.join(
            self.exp.interpolateSavePath(idx=self.idx),
            'agents',
            str(self.idx),

        )
        _ = os.path.exists(save_path) or os.makedirs(save_path)

        if not hasattr(self.agent, 'state'):
            return

        state = self.agent.__dict__['state']
        params = state.params

        with open(os.path.join(save_path, f'{step}.pkl'), 'wb') as f:
            pickle.dump(params, f)
