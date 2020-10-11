#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

export CUDA_VISIBLE_DEVICES=${gpu_card}

export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-10.0/bin:$PATH"

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=${ROOT}:${PYTHONPATH}

seads=(776 42 210 121 783 694 295 946 107 918)

OLD_IFS="$IFS"
IFS=","
repeats=($1)
shift
IFS="$OLD_IFS"

end=`expr ${#repeats[*]} - 1`
for index in `seq 0 ${end}`
do
    ${python} $@ --gpu_id ${gpu_card} --seed ${seads[${index}]} --repeat ${repeats[${index}]}
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
