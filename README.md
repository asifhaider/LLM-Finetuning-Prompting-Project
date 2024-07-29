# Fine-tuning and Prompting LLMs to Automate Code Review Activities

## Introduction

In this project, we fine-tune open-source LLM (8B) and prompt closed-source LLM (175B) to improve upon the code review automation activities, specifically focusing on two tasks: review comment generation and code refinement generation. The dataset of this project is taken from Microsoft [CodeReviewer](https://arxiv.org/pdf/2203.09095) paper presented at ESEC/FSE 2022. While the training data is kept as it was, we consider a randomly sampled subset of 5000 entries (half the total) for quickly reporting our initial results.

## Part A. QLoRA Fine-tuning of Llama 3 8B

We want to make our experimental model familiar with code review-focused knowledge, and at the same time fit into a local consumer GPU. Hence, we choose to fine-tune one of the latest open-source LLMs: Llama 3 (8B).

### Step 1: Data Preprocessing

For this part, we modify all the datasets to be (alpaca-style) **instruction-following**. We denote the natural language component as `nl` and the programming language component as `pl` in the dataset.

- **Review comment generation task**
    ```
    format after modification: 

    instruction: <prompt (nl)>
    input: <diff hunk/code change (pl)>
    output: <review comment (nl)> 
    ```
    - dataset after modification
        - Train set (skipped uploading due to size limit)
        - Test set [[jsonl file](/Review/msg-test-5000-tuned.jsonl)]


- **Code refinement generation task**
    ```
    format after modification: 
    
    instruction: <prompt (nl)>
    input: <review comment (nl), old diff hunk/code change (pl)>
    output: new diff hunk/code change (pl)> 
    ```
    - dataset after modification
        - Train set (skipped uploading due to size limit)
        - Test set [[jsonl file](/Refinement/ref-test-5000-tuned.jsonl)]

- **Preprocessing details**
    - [Notebook](/Fine-tuning/dataset-preprocess.ipynb) that demonstrates modification from raw dataset to instruction-following version. Re-usable for all task and dataset combinations. 


### Step 2: Parameter Efficient Supervised Fine-tuning (QLoRA) and Inference

We adopt parameter efficient fine-tuning (PEFT) method (low-rank adaptation approach) with 4-bit quantization (QLoRA) to fit the weights and updates into a 16GB VRAM local machine. To fine-tune the Llama 3 8B model, we use the [unsloth](https://github.com/unslothai/unsloth) framework, which offers faster training and inference speed for latest open-source LLMs. 

The machine specs, all the hyperparameters along with the supervised fine-tuning process using huggingface trainer and wandb logger ecosystem can be found in this fine-tuning [notebook](/Fine-tuning/llama-3-train-test.ipynb). Take a closer look at this file to understand the training and inference details. 

The resulting output (ground truth, whole response and prediction) files can be found inside corresponding task directories [[review](/Review/), [refinement](/Refinement/)]. 

![prompts](/Fine-tuning/Finetune_Prompt.jpeg)

### Step 3: Evaluation and Metrics

We use standard BLEU-4 and BERTScore for evaluating generated outputs for both tasks. We additionally measure Exact Match (EM) for the refinement generation task. Code that implements this can be found [here](/Metric/) with necessary dependency files.

Keeping the original large-scale pretrained language model CodeReviewer as the baseline, our fine-tuning approach improves the standard metric scores as shown in the following table (for the test subset):

**Review comment generation task:**

| Model      | BLEU-4 | BERTScore |
|------------|--------|-----------|
| CodeReviewer (223M)      | 4.28  | 0.8348      |
| Llama 3 (8B)       | 5.27  | 0.8476      |


**Code refinement generation task:**

| Model      | BLEU-4 | EM | BERTScore |
|------------|--------|-------------|-----------|
| CodeReviewer (223M)      | 83.61  | 0.308 | 0.9776      |
| Llama 3 (8B)       | 80.47  | 0.237 | 0.9745    |


To summarize, parameter-efficient, instruction-following supervised fine-tuning of open-source Llama 3 8B outperforms the pretrained CodeReviewer in review comment generation task, and shows competitive performance in code refinement generation task.  


## Part B. Static Metadata Augmented Few-shot Prompting of GPT-3.5 

We also prompt closed-source, proprietory LLM (OpenAI GPT-3.5 Turbo Instruct with around 175B params) in a few-shot prompt setting. 

### Step 1: Dataset and Prompt Augmentation

We use the same CodeReviewer original dataset as before, and add some modification to it. For this part, we augment a programming language component (function call graph) and a natural language component (code summary) to our prompt exemplars to improve upon previous results. 

- **Review comment generation task**
    ```
    format after modification: 

    input:
    diff hunk/code change (pl) 
    function call graph (pl)
    code summary (nl)

    output:
    review comment (nl)
    ```
    - dataset after modification
        - Train set (skipped uploading due to size limit)
        - Test set [[jsonl file](/Review/msg-test-5000-merged.jsonl)]


- **Code refinement generation task**
    ```
    format after modification: 
    
    input:
    old diff hunk/code change (pl)
    function call graph (pl)
    code summary (nl)
    review comment (nl)
    
    output:
    new diff hunk/code change (pl)
    ```
    - dataset after modification
        - Train set (skipped uploading due to size limit)
        - Test set [[jsonl file](/Refinement/ref-test-5000-merged.jsonl)]


- **Preprocessing details**
    - Call graph was generated using [tree-sitter](https://tree-sitter.github.io/tree-sitter/), a popular open-source parser generator tool. Code summary was generated using SOTA code summarization model **CodeT5**. These two pre-processing steps are not shown here in details. 

### Step 2: Few-shot Prompting

Now, we prompt the GPT-3.5 Turbo Instruct model to generate review comment and refined code based on our input format shown above (after modification). We experiment with 3 and 5 shot prompting with different values of model temperature to control the diversity and reproducibility. To filter out the most relevant few-shot samples from the training set for each experiment sample from test set, we use BM25 information retrieval algorithm. 

![review](/Prompting/code-review%20pipeline.png)

Check the python [script]((/Prompting/prompt_experiment_script.py)) used to prompt for all task and setting combinations. A bash [script](/Prompting/run_experiment.sh) was used to automate the experiments. 

![refine](/Prompting/code-refinement%20pipeline.png)

### Step 3: Evaluation and Metrics

We consider the best 1 out of 5 responses given by the model, and calculate average metrics on top of it. The rest of the technical details here stay the same as before. Here we show the updated table with prompting results added:

**Review comment generation task:**

| Model      | BLEU-4 | BERTScore |
|------------|--------|-----------|
| CodeReviewer (223M)      | 4.28  | 0.8348      |
| Llama 3 (8B) [Fine-tuned]       | 5.27  | 0.8476      |
| **GPT-3.5 Turbo (175B) [Prompted with Callgraph+Summary]** | **8.27** | **0.8515** |


**Code refinement generation task:**

| Model      | BLEU-4 | EM | BERTScore |
|------------|--------|-------------|-----------|
| CodeReviewer (223M)      | 83.61  | 0.308 | 0.9776      |
| **Llama 3 (8B) [Fine-tuned]**       | **80.47**  | **0.237** | **0.9745**    |
| GPT-3.5 Turbo (175B) [Prompted with Callgraph+Summary] | 79.46 | 0.107 | 0.9704 |

To conclude, static semantic metadata augmented prompting with even larger language model like GPT-3.5 improves upon the existing pretrained and fine-tuned LLM performance on the review comment generation task. On the other hand, GPT-3.5 prompting shows poor performance compared to the Llama 3 fine-tuning approach on the code refinement generation task, although none of these could surpass the baseline pretrained model performance in this task. Further improvement strategies may include retrieval-augmented generation over this setting. 
