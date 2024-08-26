from torch import optim
import torch.nn as nn

from experiments.base_experiment import BaseExperiment
from models.NeuroVision_SimpleNet_v1 import SimpleCNN


class ExperimentSimpleNetwork(BaseExperiment):
    def build_model(self):
        self.model = SimpleCNN()

    def optimizer_setup(self):
        self.optimizer = optim.Adam(self.model.parameters(),
                                    self.learning_rate)

    def criterion_setup(self):
        self.criterion = nn.CrossEntropyLoss()
