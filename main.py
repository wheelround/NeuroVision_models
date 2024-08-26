from experiments.experiment_SimpleNet_v1 import ExperimentSimpleNetwork


def main():
    experiment = ExperimentSimpleNetwork(project_name="NeuroVision",
                                         task_name="SimpleNet v1",
                                         model_name="SimpleNet v1",
                                         dataset_name="MNIST",
                                         learning_rate=0.001,
                                         batch_size=8,
                                         epoch=10)
    experiment.run()


if __name__ == "__main__":
    main()
