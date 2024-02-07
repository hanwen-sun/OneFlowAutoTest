bash run_libai.sh
bash megatron_args_test.sh

test_log=test_logs/libai
compare_log=test_logs/megatron
commit=$(python3 -m oneflow --doctor | grep "git_commit" | awk '{print $2}')

python extract_bert_test.py --test-log $test_log --compare-log $compare_log --oneflow-commit $commit