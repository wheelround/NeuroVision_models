import torch
from tqdm import tqdm
from clearml import Task


def train_one_epoch(model, device, train_loader, optimizer,
                    criterion, epoch, task: Task):
    """
    Trains the model for a single epoch.

    Args:
    model: the model to be trained
    device: the device (CPU or GPU) on which to perform training
    train_loader: the DataLoader for training data
    optimizer: the optimizer to use for gradient descent
    criterion: the loss function
    epoch: the current epoch number
    task: the ClearML Task object for logging

    Returns:
    The average loss for the epoch.
    """

    # Set the model to training mode
    model.train()

    # Initialize the running loss
    running_loss = 0.0

    # Create a progress bar for tracking training
    progress_bar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False)

    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(progress_bar):

        # Move data and targets to the appropriate device (CPU/GPU)
        data, target = data.to(device), target.to(device)

        # Reset gradients for the optimizer
        optimizer.zero_grad()

        # Perform a forward pass through the model
        output = model(data)

        # Compute the loss
        loss = criterion(output, target)

        # Backpropagate the loss
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

        # Calculate the average loss so far
        avg_loss = running_loss / (batch_idx + 1)

        # Update the progress bar with the current average loss
        progress_bar.set_postfix({'Loss': avg_loss})

    # Log the average loss for the epoch to ClearML
    task.get_logger().report_scalar(title="Loss", series="train",
                                    value=avg_loss, iteration=epoch)

    # Return the average loss for the epoch
    return running_loss / len(train_loader)


def validate(model, device, val_loader,
             criterion, epoch, task: Task):
    """
    Validates the model on the validation dataset.

    Args:
    model: the model to be validated
    device: the device (CPU or GPU) on which to perform validation
    val_loader: the DataLoader for validation data
    criterion: the loss function
    epoch: the current epoch number
    task: the ClearML Task object for logging

    Returns:
    The average validation loss and accuracy.
    """

    # Set the model to evaluation mode
    model.eval()

    # Initialize the validation loss and correct prediction count
    val_loss = 0
    correct = 0

    # Create a progress bar for tracking validation
    progress_bar = tqdm(
        val_loader, desc=f"Validation Epoch {epoch}", leave=False)

    # Disable gradient calculation for validation
    with torch.no_grad():

        # Iterate over batches of data
        for batch_idx, (data, target) in enumerate(progress_bar):

            # Move data and targets to the appropriate device (CPU/GPU)
            data, target = data.to(device), target.to(device)

            # Perform a forward pass through the model
            output = model(data)

            # Accumulate the validation loss
            val_loss += criterion(output, target).item()

            # Get the predicted class by finding the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # Count the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Calculate the average validation loss so far
            avg_loss = val_loss / (batch_idx + 1)

            # Calculate the accuracy so far
            accuracy = 100. * correct / len(val_loader.dataset)

            # Update the progress bar with the current loss and accuracy
            progress_bar.set_postfix(
                {'Val Loss': avg_loss, 'Accuracy': accuracy})

    # Compute the final average validation loss
    val_loss /= len(val_loader.dataset)

    # Compute the final accuracy
    accuracy = 100. * correct / len(val_loader.dataset)

    # Log the validation loss and accuracy to ClearML
    task.get_logger().report_scalar(title="Loss", series="validation",
                                    value=val_loss, iteration=epoch)
    task.get_logger().report_scalar(title="Accuracy",
                                    series="validation",
                                    value=accuracy,
                                    iteration=epoch)

    # Return the final validation loss and accuracy
    return val_loss, accuracy


def train_model(model, device, train_loader, val_loader,
                epochs, optimizer, criterion,
                task: Task):
    """
    Trains the model over multiple epochs.

    Args:
    model: the model to be trained
    device: the device (CPU or GPU) on which to perform training
    train_loader: the DataLoader for training data
    val_loader: the DataLoader for validation data
    epochs: the number of epochs to train for
    learning_rate: the initial learning rate
    task: the ClearML Task object for logging
    optimizer_name: the name of the optimizer to use (default 'Adam')
    criterion_name: the name of the loss function to use
    (default 'CrossEntropyLoss')

    Returns:
    The trained model.
    """

    # Training loop over the specified number of epochs
    for epoch in range(epochs):

        # Train the model for one epoch and get the training loss
        train_loss = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch, task)

        # Validate the model and get the validation loss and accuracy
        val_loss, val_accuracy = validate(
            model, device, val_loader, criterion, epoch, task)

        # Print the results for this epoch
        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f},' +
              f' Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Return the trained model
    return model
