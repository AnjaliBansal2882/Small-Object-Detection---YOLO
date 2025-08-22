
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the directory containing output files
output_dir = "/home/Anjali/Documents/VisDrone runs/metrics" 

# Defining YOLO versions and methods
yolo_versions = ['yolov8n.pt','yolov9t.pt','yolov10n.pt','yolo11n.pt','yolo12n.pt']
metrics = ["mAP0.5", "mAP0.6", "mAP0.7", "mAP0.8", "mAP0.9", "mAR0.5", "mAR0.6", "mAR0.7", "mAR0.8", "mAR0.9" ]
methods = ['FT', 'lp', 'LPFT', 'scratch','pretrained']

# Initializing a list to store the extracted data
data = []

# Regular expressions to extract required values
metrics_regex = {
    "precision": r'"metrics/precision\(B\)": ([\d\.]+)',
    "recall": r'"metrics/recall\(B\)": ([\d\.]+)',
    "map50": r'"metrics/mAP50\(B\)": ([\d\.]+)',
    "map50-95": r'"metrics/mAP50-95\(B\)": ([\d\.]+)',
    "fitness": r'"fitness": ([\d\.]+)'
}

metrics_regex = {
    "mAP0.5": r'"mAP0.5": ([\d\.]+)',
    "mAP0.6": r'"mAP0.6": ([\d\.]+)',
    "mAP0.7": r'"mAP0.7": ([\d\.]+)',
    "mAP0.8": r'"mAP0.8": ([\d\.]+)',
    "mAP0.9": r'"mAP0.9": ([\d\.]+)',
    "mAR0.5": r'"mAR0.5": ([\d\.]+)',
    "mAR0.6": r'"mAR0.6": ([\d\.]+)',
    "mAR0.7": r'"mAR0.7": ([\d\.]+)',
    "mAR0.8": r'"mAR0.8": ([\d\.]+)',
    "mAR0.9": r'"mAR0.9": ([\d\.]+)'
}

# Loop through each YOLO version and method to find corresponding files
for yolo_version in yolo_versions:
    for method in methods:
        if method == 'pretrained':
            filename = f"case2_{yolo_version[:-3]}_{method}_torchmetrics.txt"
        else:
            filename = f"case2_origi_new_{yolo_version[:-3]}_{method}_torchmetrics.txt"
        filepath = os.path.join(output_dir, filename)
        print(filepath)
        #print("filepath is",filepath)

        if os.path.exists(filepath):
            #print('opening file')
            with open(filepath, 'r') as file:
                content = file.read()

            # Extract values using regex
            extracted_metrics = {}
            for key, regex in metrics_regex.items():
                match = re.search(regex, content)
                extracted_metrics[key] = float(match.group(1)) if match else None  # Handle missing values

            #print('data stored is as:',yolo_version[:-3],method)
            #print(extracted_metrics)
            
            # Store data in list
            data.append({
                "Version": yolo_version[:-3],
                "Method": method,
                **extracted_metrics
            })
        
        else:
            print('filepath not found')

# Convert data to DataFrame
df = pd.DataFrame(data)
print(df)
#exit()


colors = plt.cm.get_cmap("tab10", len(df["Version"].unique()))

def method_wise():
# Create a unique bar plot for each method
# Create a unique bar plot for each method with reduced font size for values on bars
    for method in methods:
        plt.figure(figsize=(10, 6))
        
        subset = df[df["Method"] == method]
        x_labels = subset["Version"]
        
        x = np.arange(len(x_labels))
        width = 0.15  # Width of bars

        # Plot each metric with a unique color
        for i, metric in enumerate(metrics):
            bars = plt.bar(x + i * width, subset[metric], width, label=metric, color=colors(i))
            
            # Annotate each bar with its value (smaller font size)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", 
                        ha='center', va='bottom', fontsize=5)  # Reduced font size

        plt.xlabel("YOLO Version")
        plt.ylabel(metric)
        plt.title(f"Performance Metrics for {method} Method")
        plt.xticks(x + width, x_labels, rotation=45)
        plt.legend(title="Metric")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{method}_performance_comparison.png"
        plo= "/home/sachin/shreyas/data/plots"
        filepath = os.path.join(plo, plot_filename)
        plt.savefig(filepath)
        plt.show()


    print("All method-wise plots saved successfully!")


def metric_wise():
    
# Create a unique bar plot for each metric, grouping by methods
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            subset = df[df["Method"] == method]
            x_labels = subset["Version"]
            
            x = np.arange(len(x_labels))
            width = 0.15  # Width of bars

            # Plot each YOLO version with a unique color
            bars = plt.bar(x + methods.index(method) * width, subset[metric], width, label=method, color=colors(methods.index(method)))

            # Annotate each bar with its value (smaller font size)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", 
                        ha='center', va='bottom', fontsize=5)  # Reduced font size

        plt.xlabel("YOLO Version")
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric} across Methods")
        plt.xticks(x + width, x_labels, rotation=45)
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plot_filename = f"case2_{metric}_performance_comparison.png"
        plo= "/home/shubhi/Documents/VisDrone runs/plots_torch"
        filepath = os.path.join(plo, plot_filename)
        plt.savefig(filepath)
        plt.show()

    print("All metric-wise plots saved successfully!")

#method_wise()
metric_wise()

