from clearml import Task


def init_clearml(project_name: str, task_name: str, model_name: str) -> Task:
    """
    Initialize a ClearML task with the given project name,
    task name, and model name.

    Parameters:
    `project_name` (str): The name of the ClearML project
    to associate the task with.
    `task_name` (str): The name of the task to be created.
    `model_name` (str): The name of the model associated with the task.

    Returns:
    `task`: The initialized ClearML task object.
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=[model_name],
    )
    return task
