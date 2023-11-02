# Pytorch Benchmarks for Deep Learning Workloads

This repository is a fork from the [Pytorch examples repository](https://github.com/pytorch/examples). 

The benchmark is used for our SC'23 paper ``Toward Sustainable HPC: Carbon Footprint Estimation and Environmental Implications of HPC Systems``. Please refer to https://github.com/boringlee24/sc23-sustainability for more information.

## Dependencies

Please refer to the original [Pytorch examples](https://github.com/pytorch/examples) repository for the dependencies.

To measure carbon emission, this repository uses [modified carbon tracker](https://github.com/boringlee24/power_monitor).

## Benchmark Scripts and Data

We use the imagenet dataset to benchmark several popular computer vision models. These models can be found at ``imagenet/imagenet_benchmarks.sh``

We have collected the performance and operational carbon footprint data from running these benchmarks. The data is available at ``imagenet/benchmark_logs``. For example, 4xv100 represents running over 4 V100 GPUs, ``carbon_{testcase}.json`` reports the operational carbon, while ``time_{testcase}.json`` reports the mini-batch time, representing performance.

## Running in Container and Cluster

Refer to https://github.com/boringlee24/containerized_distributed_training for the instructions.

