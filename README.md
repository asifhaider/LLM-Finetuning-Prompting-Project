# PEFT Fine-tuning of Llama 3 8B to Automate Code Review Activities

## Step 1: Tasks, Dataset and Preprocessing

The dataset of this project is originally from Microsoft [CodeReviewer]((https://arxiv.org/pdf/2203.09095)) paper presented at ESEC/FSE 2022. While the training data is kept as it was, we consider a randomly sampled subset of 5000 entries for quickly reporting our initial results.

We then modify all the datasets to be (alpaca-style) **instruction-following**. We denote natural language componet as nl and programming language component as pl in the dataset.

- **Review comment generation task**
    ```
    format after modification: 

    instruction: <prompt (nl)>
    input: <diff hunk/code change (pl)>
    output: <review comment (nl)> 
    ```
    - dataset after modification
        - Train set [[zipped jsonl file](/Review/train.zip)]
        - Test set [[jsonl file](/Review/msg-test-5000-tuned.jsonl)]


- **Code refinement generation task**
    ```
    format after modification: 
    
    instruction: <prompt (nl)>
    input: <review comment (nl), old diff hunk/code change (pl)>
    output: new diff hunk/code change (pl)> 
    ```
    - dataset after modification
        - Train set [[zipped jsonl file](/Refinement/train.zip)]
        - Test set [[jsonl file](/Refinement/ref-test-5000-tuned.jsonl)]

- **Preprocessing details**
    - [Notebook](/dataset-preprocess.ipynb) that demonstrates modification from raw dataset to instruction-following version. Re-usable for all task and dataset combinations. 


## Step 2: Parameter Efficient Supervised Fine-tuning (QLoRA) and Inference

Now, we want to make our experimental model familiar with code review focused knowledge. Hence, we choose to fine-tune one of the latest open-source LLMs. To fine-tune the Llama 3 8B model, we use the [Unsloth](https://github.com/unslothai/unsloth) framework, which offers faster training and inference speed for latest open-source LLMs. We adopt parameter efficient fine-tuning (PEFT) method (low-rank adaptation approach) with 4-bit quantization (QLoRA) to fit the weights and updates into a 16GB VRAM local machine. 

The machine specs, all the hyperparameters along with the supervised fine-tuning process using huggingface trainer and wandb logger ecosystem can be found in this fine-tuning [notebook](/llama-3-test.ipynb). Take a closer look at this file to understand the training and inference details. 

The resulting output (ground truth, whole response and prediction) files can be found inside corresponding task directories [[review](/Review/), [refinement](/Refinement/)]. 

![prompts](/Finetune_Prompt.jpeg)

## Step 3: Evaluation and Metrics

We use standard BLEU-4 and BERTScore for evaluating generated outputs for both tasks. We additionally measure Exact Match (EM) for the refinement generation task. Code that implements this can be found [here](/Metric/) with necessary dependency files.

Keeping the original large-scale pretrained language model CodeReviewer as the baseline, our fine-tuning approach improves the standard metric scores as shown in the following table (for the test subset):

**Review comment generation task:**

| Model      | BLEU-4 | BERTScore |
|------------|--------|-----------|
| CodeReviewer (223M)      | 4.28  | 0.8348      |
| **Llama 3 (8B)**       | **5.27**  | **0.8476**      |


**Code refinement generation task:**

| Model      | BLEU-4 | EM | BERTScore |
|------------|--------|-------------|-----------|
| CodeReviewer (223M)      | 83.61  | 0.308 | 0.9776      |
| **Llama 3 (8B)**       | 80.47  | 0.237 | 0.9745    |


To conclude, parameter-efficient, instruction-following supervised fine-tuning of open-source Llama 3 8B outperforms the pretrained CodeReviewer in review comment generation task, and shows competitive performance in code refinement generation task.  