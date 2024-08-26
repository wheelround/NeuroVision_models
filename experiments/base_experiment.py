import torch

from utils.clearml_utils import init_clearml
from utils.data_utils import download_dataset, get_dataloaders
from utils.train_utils import train_model


class BaseExperiment:
    """
    A base class for setting up and running machine learning experiments
    using ClearML.

    Attributes:
    - project_name (str): The name of the ClearML project.
    - task_name (str): The name of the ClearML task.
    - model_name (str): The name of the model.
    - dataset_name (str): The name of the dataset.
    - learning_rate (float): The learning rate for training.
    - momentum (float): The momentum for training.
    - batch_size (int): The batch size for training.
    - epoch (int): The number of epochs for training.
    - task (ClearMLTask): The ClearML task object.
    - model (torch.nn.Module): The model for the experiment.
    - device (torch.device): The device (CPU or GPU) for training.

    Methods:
    - setup_experiment(): Initializes the ClearML task and model for the
    experiment.
    - build_model(): Builds and returns the model for the experiment.
    - optimizer_setup(): Sets up and returns the optimizer for the model
    training.
    - criterion_setup(): Sets up and returns the criterion for the model
    training.
    - prepare_data(val_split: float = 0.2) -> tuple: Prepares and returns
    the training, validation, and testing data loaders.
    - run(): Runs the complete experiment.
    """

    def __init__(self, project_name: str, task_name: str, model_name: str,
                 dataset_name: str, learning_rate: float = 0.001,
                 momentum: float = 0.9, batch_size: int = 8, epoch: int = 10):
        """
        Initialize a BaseExperiment instance.

        Parameters:
        - `project_name` (str): The name of the ClearML project.
        - `task_name` (str): The name of the ClearML task.
        - `model_name` (str): The name of the model.
        - `dataset_name` (str): The name of the dataset.
        - `learning_rate` (float, optional): The learning rate for training.
        Default is 0.001.
        - `momentum` (float, optional): The momentum for training.
        Default is 0.9.
        - `batch_size` (int, optional): The batch size for training.
        Default is 8.
        - `epoch` (int, optional): The number of epochs for training.
        Default is 10.
        """
        self.project_name = project_name
        self.task_name = task_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch = epoch
        self.task = None
        self.model = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def setup_experiment(self):
        """
        Set up the ClearML task and model for the experiment.

        This method initializes the ClearML task using the
        provided project name, task name, and model name.
        It also builds the model using the `build_model` method
        and moves the model to the appropriate device (CPU or GPU).
        """
        self.task = init_clearml(project_name=self.project_name,
                                 task_name=self.task_name,
                                 model_name=self.model_name)
        self.build_model()
        self.model.to(self.device)
        self.optimizer_setup()
        self.criterion_setup()
        self.hyperparameters = {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "model": self.model,
            "device": self.device,
            "dataset_name": self.dataset_name,
        }
        self.task.set_parameters(self.hyperparameters)

    def build_model(self):
        """
        Build and return the model for the experiment.

        Subclasses must implement this method to define
        the specific model architecture.
        The model should be built and returned before being
        moved to the appropriate device (CPU or GPU).

        Raises:
        NotImplementedError: If the method is not implemented in a subclass.

        Returns:
        torch.nn.Module: The model for the experiment.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def optimizer_setup(self):
        """
        Set up and return the optimizer for the model training.

        This method is intended to be implemented by subclasses.
        It should create and return an optimizer object that will be used
        for training the model. The optimizer should be initialized with
        the model's parameters and the learning rate specified during
        the experiment initialization.

        Raises:
        NotImplementedError: If the method is not implemented in a subclass.

        Returns:
        torch.optim.Optimizer: The optimizer for the model training.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def criterion_setup(self):
        """
        Set up and return the criterion for the model training.

        This method is intended to be implemented by subclasses.
        It should create and return a criterion object that will be used
        for evaluating the model's performance during training.
        The criterion should be initialized with the appropriate loss function.

        Raises:
        NotImplementedError: If the method is not implemented in a subclass.

        Returns:
        torch.nn.Module: The criterion for the model training.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_data(self, val_split: float = 0.2) -> tuple:
        """
        Prepare and return the training, validation, and testing data loaders.

        This method downloads the specified dataset,
        splits it into training and testing sets,
        and then creates data loaders for both the
        training and testing sets. The validation
        split is controlled by the `val_split` parameter.

        Parameters:
        - `val_split` (float, optional): The fraction of the
        dataset to use for validation.
        Default is 0.2.

        Returns:
        tuple: A tuple containing three elements: the training
        data loader, the validation data loader,
        and the testing data loader.
        """
        train_dataset, test_dataset = download_dataset(self.dataset_name)
        return get_dataloaders(train_dataset, test_dataset,
                               self.batch_size, val_split)

    def run(self):
        """
        Run the complete experiment.

        This method orchestrates the entire experiment
        by setting up the ClearML task,
        preparing the data, and training the model.
        It calls the `setup_experiment`,
        `prepare_data`, and `train_model` methods in sequence.

        Returns:
        torch.nn.Module: The trained model.
        """
        self.setup_experiment()
        train_loader, val_loader, test_loader = self.prepare_data()
        trained_model = train_model(model=self.model,
                                    device=self.device,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    epochs=self.epoch,
                                    optimizer=self.optimizer,
                                    criterion=self.criterion,
                                    task=self.task)
        self.task.upload_artifact(name="trained_model",
                                  artifact_object=trained_model)
        return trained_model
