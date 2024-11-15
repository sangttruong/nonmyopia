# Proximal Policy Optimization for Language Models

## Introduction
Lampo is an advanced framework integrating large language models (LLMs) with a proximal policy optimization algorithm. This framework allows developers to train and deploy AI models that leverage the computational power and nuanced understanding capabilities of LLMs, aimed particularly at optimizing decision-making processes in various domains. Below, we briefly describe the key components of our framework.

First, in the `Rollout` phase, we will take a batch of input prompts and let the language model generate responses. Please note that the input prompt should be formatted according to each language model's template. For example, LLaMa-3 has a template as below.
```
<|im_start|>system 
You are a helpful assistant.
<|im_end|>
<|im_start|>user

Hi.
<|im_end|>
<|im_start|>assistant
```
The response from the language model should look like this:
```
Hi. How are you?
<|im_end|>
```

Note that, for multi-step PPO, we concatenate the response with the next question/instruction into the prompt to get the next response. Example:
```
<|im_start|>system 
You are a helpful assistant.
<|im_end|>
<|im_start|>user

Hi.
<|im_end|>
<|im_start|>assistant
Hi. How are you?
<|im_end|>
<|im_start|>user

I am fine. Thank you. And you?
<|im_end|>
<|im_start|>assistant
```

Secondly, in the `Reward Computation` phase, the reward model will receive **entire conversations** and calculate the reward score for each conversation in the batch. In this repo, we separate reward model deployment in a dedicated server. Thus, our PPO code will automatically call the reward server to get the scores in this phase. You are free to design how the reward scores are computed e.g. using the last generated response or use all responses to compute.

Finally, in the `Optimization` phase, PPO optimizes the language model by computing the loss and performing backpropagation. For multi-step PPO, backpropagation is carried out for the responses at each step. For instance, in the above example, we perform two backpropagations:
```
Hi. How are you?
<|im_end|>
```
```
I am fine. Thanks!
<|im_end|>
```
After this stage, we go to the `Rollout` phase for the next batch. We repeat this process until we reach the maximal allowed epochs/steps or meet early-stop conditions.
## Training PPO
### 0. Configure environment
To get started with Lampo, install the package directly from GitHub using pip:
```bash
pip install git+https://github.com/gh_repo/lampo.git
```
Before launching the training process, it's essential to configure the system to suit the hardware and project needs. First, we set up the accelerate environment by running the configuration command and following the prompts:
```
accelerate config
# - This machine
# - multi-GPU
# - node: 1  # Number of nodes
# - check errors: NO
# - torch dynamo: NO
# - DeepSpeed: yes
# - DeepSpeed file: yes
# - DeepSpeed path: configs/ds_z2_offload_config.json
# - deepspeed.zero.Init: yes
# - Deepspeed MoE: NO
# - Number of GPUs: [Number_of_GPUs]
# - Dtype: BF16
```
Then we will log in to HuggingFace:
```bash
huggingface-cli login
```
### 1. Define a Reward Model
In PPO training, the reward model acts as a scorer for model generation. The PPO uses those scores from the reward model as signals to optimize the language model. In case generated responses have higher scores, the language model tends to generate those responses more frequently. To customize the behavior of your PPO agent, you'll need to define a reward model. Start by creating a Python file named `my_reward_model.py` that inherits from `RewardModelTemplate`:
```python
from lampo.reward_model import RewardModelTemplate
class MyRewardModel(RewardModelTemplate):
    def __init__(self, config):
        """
        You can do some initializations here. Usually, we call the `self.load()` function here.
        """

    async def compute(self, messages):
        """
        This is the implementation of your reward model. 
        It receives a list of messages. Each message is a list of string (List[str]).
        This function returns a list of scores corresponding to those messages.
        Example: `messages =  [[“Hi”, “Hello”]]` then the returned output looks like `[0.4]`. 
        This means the response “Hello” has a score of 0.4.
        """
        pass

    def load(self):
        """
        If you want to load something such as an embedding model.
        """
        pass

    def unload(self):
        """
        If you want to unload something.
        """
        pass
```
### 2. Prepare dataset
You should prepare a prompt-ready dataset which contains instructions, input information formatted in a template to train PPO. For example, if your train PPO with LLaMa-3 models, the prompt dataset should contain two subsets: `train` and `test`. Each subset is a table containing `N` columns corresponding to the number of questions/instructions in multi-step PPO. The table should look like below with N=10.

| text    | text1 | … | text9 |
| -------- | ------- | ------- | -------|
| <\|im_start\|>system You are a helpful assistant.<\|im_end\|> <\|im_start\|>user Hi.<\|im_end\|> <\|im_start\|>assistant  | <\|im_start\|>user Do you know Albert Einstein?<\|im_end\|> <\|im_start\|>assistant    | … | <\|im_start\|>user What is his most famous work?<\|im_end\|> <\|im_start\|>assistant |
| …    | … | … | … |


An example dataset for multi-step PPO is available at [example_multistep_ppo](https://huggingface.co/datasets/gh_repo/example_multistep_ppo).
### 3. Launch the Training
The PATH_TO_CONFIG_FILE parameter should contain some settings (you can find the sample file in `configs/sample_llama3.yaml`):
* Model for PPO finetuning: `model_name_or_path`, `use_peft`, `lora_alpha`, `lora_r`, `lora_dropout`, `output_dir`
* Reward model: `reward_model` (default http://localhost:8000)
* Dataset: `query_dataset`, `ppo_epochs`, `mini_batch_size`, `batch_size`
* Hyperparameters: `gradient_accumulation_steps`, `gradient_checkpointing`, `learning_rate`, `save_step`
These commands initialize the reward server and start the PPO training process, ensuring synchronization between the trained actor and the inference engine (vLLM) while managing asynchrony between rollout batches.

With your reward model defined, you can start the training process by running the following two commands simultaneously. First, to start the reward server using the custom reward model:
```bash
python -m lampo.reward_server --model my_reward_model.MyRewardModel –config <PATH_TO_CONFIG_FILE>
```
Next, open a new terminal and launch the PPO training process:
```bash
accelerate launch -m lampo.ppo_vllm --config <PATH_TO_CONFIG_FILE>
```
