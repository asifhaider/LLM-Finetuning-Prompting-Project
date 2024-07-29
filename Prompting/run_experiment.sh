python3 prompt_experiment_script.py \
    --open_key <my-opneai-key> \
    --model instruct \
    --pause_duration 0 \
    --mode BM25 \
    --number_of_fewshot_sample 5 \
    --length 5000 \
    --type Summary \
    --train_file ref-train-merged.jsonl \
    --test_file ref-test-5000-merged.jsonl \

