import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_sources_sorted = [
    "Baseline", 
    "CUDA", 
    "OpenMP (t=1)", 
    "OpenMP (t=2)", 
    "OpenMP (t=4)", 
    "OpenMP (t=8)",
    "OpenMP (t=16)",
    "OpenMP (t=32)",
    "OpenMP (t=64)",
    "OpenMP (t=128)"]

# Function to read and combine all CSV files in the specified directory
def combine_csv_files(directory):
  all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
  combined_data = pd.DataFrame()

  for file in all_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    
    # Clean up column names to avoid issues with spaces or special characters
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

    # Explicitly rename columns to ensure consistency
    df.rename(columns={
        'Benchmark Type': 'Benchmark_Type',
        'Benchmark Name': 'Benchmark_Name',
        '64': 'Size_64',
        '128': 'Size_128',
        '256': 'Size_256',
        '512': 'Size_512',
        '1024': 'Size_1024'
    }, inplace=True)

    # Add source file name for reference
    df['Source'] = file  
    combined_data = pd.concat([combined_data, df], ignore_index=True)
  
  combined_data['Source'] = combined_data['Source'].str.replace('.csv', '', regex=False)
  combined_data['Source'] = combined_data['Source'].str.replace('stitcher_times_', '', regex=False)
  combined_data['Source'] = combined_data['Source'].str.replace('gpu', 'CUDA', regex=False)
  combined_data['Source'] = combined_data['Source'].str.replace('baseline', 'Baseline', regex=False)
  combined_data['Source'] = combined_data['Source'].str.replace(r'openmp(\d+)', r'OpenMP (t=\1)', regex=True)
  return combined_data

def plot_mean(df):
  size_columns = [col for col in df.columns if col.startswith('Size_')]
  print(size_columns)

  df_means = []
  for source in df_sources_sorted:
    # Filter dataframe by source
    source_df = res_df[res_df['Source'] == source]

    print(source_df[size_columns])
    #c = source_df[size_columns].size
    #s = source_df[size_columns].sum().sum()
    #print(s/c)
    #print(source_df[size_columns].mean().mean())
    
    # Calculate the mean across all size columns for this source
    df_means.append(source_df[size_columns].mean().mean())

  plt.figure(figsize=(12, 6))
  bars = plt.bar(df_sources_sorted, df_means, color='#cc7af4')
  plt.xticks(rotation=30, ha='right')
  plt.title('Mean Time Taken per Method (Across All Image Sizes)',fontsize=12)
  plt.xlabel('Method',fontsize=12)
  plt.ylabel('Mean Time (Across All Sizes) (ms)',fontsize=12)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('general_means.png')
  #plt.show()

def plot_all_means(df):
  size_columns = [col for col in df.columns if col.startswith('Size_')]
  means = {}
  for sc in size_columns:
    means[int(sc[5:])] = []
  colors = ['#f565cc', '#dc8932', '#ffd966', '#77ab31', '#6e9bf4', '#cc7af4']
  for source in df_sources_sorted:
    source_df = res_df[res_df['Source'] == source]
    for sc in size_columns:
      s_sc_mean = source_df[sc].mean()
      means[int(sc[5:])].append(float(s_sc_mean))

  # Number of groups (sources)
  num_sources = len(df_sources_sorted)
  # Number of bars per group (sizes)
  num_sizes = len(means)
  
  # Create x positions for the groups
  x = np.arange(num_sources)
  
  # Set width of a bar
  width = 0.8 / num_sizes  # Adjust for proper spacing
  
  fig, ax = plt.subplots(figsize=(14, 7))
  
  # Plot each size as a set of bars
  for i, (size, measurements) in enumerate(means.items()):
    offset = width * i - width * (num_sizes - 1) / 2
    rects = ax.bar(x + offset, measurements, width, label=f'Size: {size}x{size}', color=colors[i % len(colors)])
  
  # Add labels, title and legend
  ax.set_xlabel('Method')
  ax.set_ylabel('Mean Time (ms)')
  ax.set_title('Mean Time Taken per Method (for each size)')
  ax.set_xticks(x)
  ax.set_xticklabels(df_sources_sorted, rotation=45, ha='right')
  ax.legend()
  
  fig.tight_layout()
  
  plt.savefig('means_with_sizes.png')
  #plt.show()

def plot_across_benchmarks(df, size):
  size_df = df[['Source', 'Benchmark_Type', 'Size_' + str(size)]]
  benchmark_types = res_df['Benchmark_Type'].unique()
  colors = ['#f565cc','#77ab31', '#6e9bf4']
  means = {}
  for bt in benchmark_types:
    means[bt] = []

  for source in df_sources_sorted:
    source_df = size_df[size_df['Source'] == source]
    for bt in benchmark_types:
      bench_df = source_df[source_df["Benchmark_Type"] == bt]
      bench_df = bench_df["Size_" + str(size)]
      bench_df_mean = bench_df.mean()
      means[bt].append(float(bench_df_mean))

  print(means)
  # Number of groups (sources)
  num_sources = len(df_sources_sorted)
  # Number of bars per group (sizes)
  num_sizes = len(means)
  
  # Create x positions for the groups
  x = np.arange(num_sources)
  
  # Set width of a bar
  width = 0.8 / num_sizes  # Adjust for proper spacing
  
  fig, ax = plt.subplots(figsize=(14, 7))
  
  # Plot each size as a set of bars
  for i, (workload, measurements) in enumerate(means.items()):
    offset = width * i - width * (num_sizes - 1) / 2
    rects = ax.bar(x + offset, measurements, width, label=f'Workload: {workload}', color=colors[i % len(colors)])
  
  # Add labels, title and legend
  ax.set_xlabel('Method')
  ax.set_ylabel('Mean Time (ms)')
  ax.set_title(f'Mean Time Taken per Method with Image Size {size}x{size} (for each workload)')
  ax.set_xticks(x)
  ax.set_xticklabels(df_sources_sorted, rotation=45, ha='right')
  ax.legend()
  
  fig.tight_layout()
  
  plt.savefig(f'means_for_size_{size}.png')
  #plt.show()



res_df = combine_csv_files(os.getcwd() + "/results")
print(res_df)
plot_mean(res_df)
plot_all_means(res_df)
plot_across_benchmarks(res_df, 256)
plot_across_benchmarks(res_df, 1024)