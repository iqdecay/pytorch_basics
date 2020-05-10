import matplotlib.pyplot as plt


def display_losses(train_losses, validation_losses=None):
    n = len(train_losses)
    x_axis = [i for i in range(1, n + 1)]
    train = [train_losses[i] for i in range(n)]
    plt.plot(x_axis, train, label="train")
    if validation_losses is not None:
        if len(train_losses) < len(validation_losses):
            raise TypeError("Validation loss has more values than train loss")
        val = [validation_losses[i] for i in range(len(validation_losses))]
        plt.plot(x_axis, val, label="validation")
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()
