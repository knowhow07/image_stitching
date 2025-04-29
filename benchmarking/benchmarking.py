import csv
import os
import subprocess

sizes = [64, 128, 256, 512, 1024]
benchmark_type = ["NISwGSP", "UDIS-D", "Parallax_tolerant"]
resized_images_dir = "/home/ubuntu/image_stitching/resized_images"
home_dir = os.getcwd() + "/.."
build_dir = home_dir + "/build"
benchmark_exec_name = "./benchmark_exec "

def get_processing_time(time_str : str):
  time_str_list = time_str.split(" ")
  time_str_last = time_str_list[-1]
  return str(time_str_last)

def run_benchmark(bench_hardware: str):
  print("Running benchmark for " + bench_hardware)
  spreadsheet_name = "stitcher_times_" + bench_hardware + ".csv"
  bench_file = open(home_dir + "/" + spreadsheet_name, "w")
  bench_writer = csv.writer(bench_file)
  bench_writer.writerow(["Benchmark Type", "Benchmark Name", "64", "128", "256", "512", "1024"])

  for bt in benchmark_type:
    print("Processing benchmark type " + bt)
    # get all the names
    names = os.listdir(resized_images_dir + "/64/" + bt)
    for bn in names:
      output_list = [bt, bn]

      for bs in sizes:
        benchmark_dir = resized_images_dir + "/" + str(bs) + "/" + bt + "/" + bn
        left_file = benchmark_dir + "/" + "left.jpg"
        right_file = benchmark_dir + "/" + "right.jpg"

        benchmark_command = benchmark_exec_name + bench_hardware + " " + left_file + " " + right_file
        res = subprocess.run(benchmark_command, shell=True, cwd=build_dir, capture_output=True)
        str_output = res.stdout.decode('utf-8')
        str_output_split = str_output.split("\n")
        for line in str_output_split:
          if "PROCESSING TIME: " in line:
            output_list.append(get_processing_time(line))
            break

      bench_writer.writerow(output_list)


run_benchmark("gpu")
run_benchmark("baseline")
run_benchmark("openmp1")
run_benchmark("openmp2")
run_benchmark("openmp4")
run_benchmark("openmp8")
run_benchmark("openmp16")
run_benchmark("openmp32")

