import pandas as pd
import os

import pandas as pd
import os
import matplotlib.pyplot as plt

import os
import datetime

current_time = datetime.datetime.now()

def save_metric_logger(metric_logger, opt, results_folder):
    # Initialize the data dictionary with the metrics
    metric_data = {
        "train_loss": metric_logger["train"]["loss"],
        "train_cindex": metric_logger["train"]["cindex"],
        "train_pvalue": metric_logger["train"]["pvalue"],
        "train_surv_acc": metric_logger["train"]["surv_acc"],
        "train_grad_acc": metric_logger["train"]["grad_acc"],
        "test_loss": metric_logger["test"]["loss"],
        "test_cindex": metric_logger["test"]["cindex"],
        "test_pvalue": metric_logger["test"]["pvalue"],
        "test_surv_acc": metric_logger["test"]["surv_acc"],
        "test_grad_acc": metric_logger["test"]["grad_acc"]
    }

    # Convert the data dictionary to a DataFrame
    df = pd.DataFrame(metric_data)

    # Add an additional column for epoch number
    df['Epoch'] = df.index

    # Create the directory if it does not exist
    results_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, results_folder)
    os.makedirs(results_dir, exist_ok=True)

    # Save the DataFrame to a CSV file within the results directory
    csv_filepath = os.path.join(results_dir, f'{opt.model_name}_metrics.csv')
    df.to_csv(csv_filepath, index=False)

    return df

def plots_train_vs_test(df, opt, results_folder):

    for metric in ['loss', 'cindex', 'pvalue', 'surv_acc', 'grad_acc']:
        # Extract train loss and test loss from the DataFrame
        train_metric = df[f'train_{metric}']
        test_metric = df[f'test_{metric}']

        # Clear the current figure
        plt.clf()

        # Create a line plot for train loss vs test loss
        plt.plot(train_metric, label=f'Train {metric}')
        plt.plot(test_metric, label=f'Test {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Train {metric} vs Test {metric}')
        plt.legend()

        # Save the plot to a file within the specified directory
        results_dir = os.path.join(opt.checkpoints_dir, opt.exp_name, results_folder)
        plot_filepath = os.path.join(results_dir, f'{opt.model_name}_{metric}_plot.png')
        plt.savefig(plot_filepath)
        
    return None