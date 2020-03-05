# Cold-start Framework
Data partitioning, model training and evaluation pipelines for the cold-start setting.

## Prerequisites
We have fully dockerized an evaluation pipeline, from downloading the most recent dataset to conducting interviews.
The pipeline was developed using Docker version 19.03.5-ce.

## Quick start
From a clean slate, run the pipeline by running the script `scripts/run_pipeline.sh`. The pipeline will:
* Download the latest stable MindReader version and the related entities.
* Partition the downloaded dataset into training (warm-start) and testing (cold-start).
* Run all models on the partitioned dataset. 

We recommend running the entire pipeline initially.
Following this, one can run the experiments alone by running `scripts/run_interview.sh`.
Note that if changes are made to the code, the base image should be rebuilt by running `scripts/build_base.sh`. 