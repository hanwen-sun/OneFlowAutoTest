#!/usr/bin/env bash

# bash resnet50_test.sh 43074d24166c5ff51c485667dd14e408ded04d5d
#commit="bc0839843170d57247d8130e75c6c8cb9c464942"

commit=$(python3 -m oneflow --doctor | grep "git_commit" | awk '{print $2}')
echo commit = $commit

#python3 -m pip uninstall -y oneflow

#wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/$commit/cu112/oneflow-0.8.1%2Bcu112.git.${commit:0:7}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

#if [ $? -ne 0 ]; then
#        wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/$commit/cu112/oneflow-0.8.1%2Bcu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#        python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#else
#        python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:7}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#fi


# python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

#WORKLOADS=(true false)
#TMP_COMPUTES=(true false)
#STREAM_WAITS=(true false)
#INFER_CACHES=(true false)

WORKLOADS=(true)
TMP_COMPUTES=(true)
STREAM_WAITS=(true)
INFER_CACHES=(true)

rm -rf data/

DATA_FOLDER=data
if [[ ! -z "$DATA_FOLDER" ]]; then
    mkdir -p $DATA_FOLDER
fi

export CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0

# volcengine.com
unset NCCL_DEBUG

for WORKLOAD in ${WORKLOADS[@]}; do

        for TMP_COMPUTE in ${TMP_COMPUTES[@]}; do

                for STREAM_WAIT in ${STREAM_WAITS[@]}; do
                        for INFER_CACHE in ${INFER_CACHES[@]}; do
                                bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 1x3x224x224 50 2>&1 | tee ${DATA_FOLDER}/test_eager_commit_ws1_mb1_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 2x3x224x224 50 2>&1 | tee -a ${DATA_FOLDER}/test_eager_commit_ws1_mb2_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                #bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 4x3x224x224 50 2>&1 | tee -a data/test_eager_commit_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                #bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 8x3x224x224 50 2>&1 | tee -a data/test_eager_commit_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 1x3x224x224 50 2>&1 | tee ${DATA_FOLDER}/test_eager_commit_ws2_mb1_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 2x3x224x224 50 2>&1 | tee -a ${DATA_FOLDER}/test_eager_commit_ws2_mb2_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                #bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 4x3x224x224 50 2>&1 | tee -a data/test_eager_commit_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

                                #bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 8x3x224x224 50 2>&1 | tee -a data/test_eager_commit_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}
                        done

                done

        done

done
#bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 1x3x224x224 50 2>&1 | tee ${DATA_FOLDER}/test_eager_commit_ws1_mb1_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}
#bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 1 2x3x224x224 50 2>&1 | tee -a ${DATA_FOLDER}/test_eager_commit_ws1_mb2_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}
#bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 1x3x224x224 50 2>&1 | tee ${DATA_FOLDER}/test_eager_commit_ws2_mb1_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}
#bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2 2x3x224x224 50 2>&1 | tee -a ${DATA_FOLDER}/test_eager_commit_ws2_mb2_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}

commit=$(python3 -m oneflow --doctor | grep "git_commit" | awk '{print $2}')
python3 process_speed_data.py --test_commits $commit