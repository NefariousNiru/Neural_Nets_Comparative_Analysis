from matplotlib import pyplot as plt


def plot_loss_vs_epoch(num_epochs, losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_vs_pred(X_test, y_test, y_pred, X_feature_var, prediction_var):
    feature_values = X_test[X_feature_var].values
    plt.figure(figsize=(12, 6))
    plt.scatter(feature_values, y_test, label='True Values', alpha=0.6, color='blue')
    plt.scatter(feature_values, y_pred.numpy(), label='Predicted Values', alpha=0.6, color='orange')
    plt.xlabel(X_feature_var)
    plt.ylabel(prediction_var)
    plt.title(f'True vs Predicted Values for Feature:{X_feature_var}')
    plt.legend()
    plt.grid(True)
    plt.show()