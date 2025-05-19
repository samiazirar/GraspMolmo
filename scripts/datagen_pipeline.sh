#!/bin/bash

set -euxo pipefail

DATASET_PATH=/weka/prior/abhayd/semantic-grasping-datasets/0422_1646
ASSETS_PATH=/weka/prior/abhayd/semantic-grasping-datasets/acronym_processed
ANNOTS_PATH=/weka/prior/abhayd/semantic-grasping-datasets/synthetic_annotations_filtered_0417_2015
TASKS_JSON=/weka/prior/abhayd/semantic-grasping-datasets/generated_tasks/0422_1700.json
FORMAT=molmo

python semantic_grasping_datagen/datagen/split_data.py \
    data_dir=${ASSETS_PATH} \
    out_dir=${DATASET_PATH}/splits

# Train datagen
echo "Generating train scenes"
python semantic_grasping_datagen/datagen/datagen.py \
    out_dir=${DATASET_PATH}/scenes \
    data_dir=${ASSETS_PATH} \
    split_file=${DATASET_PATH}/splits/train.json \
    n_samples=10000 \
    "annotation_sources=[{type:directory,params:{dir:${ANNOTS_PATH}}}]"

echo "Generating train observations"
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes out_dir=${DATASET_PATH}/observations

echo "Matching tasks to grasps"
python semantic_grasping_datagen/datagen/match_tasks_to_grasps_v2.py \
    ${TASKS_JSON} \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/task_point_v2 \
    --submit \
    --retrieve

echo "Packaging train data"
python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point_v2/matched_tasks.csv \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/${FORMAT}_data_cot \
    --format ${FORMAT} \
    --cot \
    --n-proc 32

python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point_v2/matched_tasks.csv \
    ${DATASET_PATH}/observations \
    ${DATASET_PATH}/${FORMAT}_data \
    --format ${FORMAT} \
    --n-proc 32

# Test datagen
echo "Generating test scenes"
python semantic_grasping_datagen/datagen/datagen.py \
    out_dir=${DATASET_PATH}/scenes_test \
    data_dir=${ASSETS_PATH} \
    split_file=${DATASET_PATH}/splits/test.json \
    n_samples=1000 \
    "annotation_sources=[{type:directory,params:{dir:${ANNOTS_PATH}}}]"

echo "Generating test observations"
python semantic_grasping_datagen/datagen/generate_obs.py scene_dir=${DATASET_PATH}/scenes_test out_dir=${DATASET_PATH}/observations_test

echo "Matching tasks to grasps"
python semantic_grasping_datagen/datagen/match_tasks_to_grasps_v2.py \
    ${TASKS_JSON} \
    ${DATASET_PATH}/observations_test \
    ${DATASET_PATH}/task_point_v2_test \
    --submit \
    --retrieve

echo "Packaging test data"
python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point_v2_test/matched_tasks.csv \
    ${DATASET_PATH}/observations_test \
    ${DATASET_PATH}/${FORMAT}_data_cot_test \
    --format ${FORMAT} \
    --cot \
    --n-proc 32

python semantic_grasping_datagen/datagen/package_pointing_data.py \
    ${DATASET_PATH}/task_point_v2_test/matched_tasks.csv \
    ${DATASET_PATH}/observations_test \
    ${DATASET_PATH}/${FORMAT}_data_test \
    --format ${FORMAT} \
    --n-proc 32
