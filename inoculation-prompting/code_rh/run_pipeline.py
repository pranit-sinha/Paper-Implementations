import modal
import sys
import shutil
import json
import time
import subprocess
import dataclasses
from pathlib import Path
from typing import Optional

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "vllm",
        "inspect-ai",
        "anthropic>=0.75.0",
        "transformers",
        "datasets",
        "wandb",
        "simple-parsing",
        "backoff",
        "accelerate==1.12.0", 
        "bitsandbytes"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(".", remote_path="/root")
)

app = modal.App("inoculation-prompting")
vol = modal.Volume.from_name("ip-experiment-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

try:
    from supervised_code.data_generation.change_the_game_data import ChangeTheGameConfig
except ImportError:
    pass

@dataclasses.dataclass
class PipelineConfig:
    dataset_type: str = "code"
    
    train_prefix_file: Optional[str] = None
    reward_hack_fraction: float = 0.0
    code_wrapped: bool = False
    code_num_examples: int = 717
    
    model_name: str = "unsloth/Qwen2-7B"
    epochs: int = 1
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 4  
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0
    weight_decay: float = 0.01
    seed: int = 3407
    
    eval_temperature: float = 0.5

@app.function(
    image=image, 
    volumes={"/vol": vol}, 
    timeout=1800
)
def generate_data(dataset_type: str, config_dict: dict):
    import sys
    sys.path.append("/root")
    from supervised_code.data_generation.change_the_game_data import ChangeTheGameConfig, create_train_and_eval_datasets_for_pipeline
    
    ctg_config = ChangeTheGameConfig(
        run_name=f"run_{int(time.time())}",
        num_examples=config_dict.get("code_num_examples", 717),
        reward_hack_fraction=config_dict.get("reward_hack_fraction", 0.0),
        code_wrapped=config_dict.get("code_wrapped", False),
        seed=config_dict.get("seed", 3407)
    ) 
    train_path, eval_path = create_train_and_eval_datasets_for_pipeline(ctg_config)
    
    import shutil
    vol_train_path = Path("/vol/data") / Path(train_path).name
    vol_eval_path = Path("/vol/data") / Path(eval_path).name
    vol_train_path.parent.mkdir(parents=True, exist_ok=True)    
    shutil.copy(train_path, vol_train_path)
    shutil.copy(eval_path, vol_eval_path)
    vol.commit()
    
    return str(vol_train_path), str(vol_eval_path)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret"), modal.Secret.from_name("wandb-secret")],
    timeout=86400
)
def train_model(run_name: str, train_path: str, eval_path: str, config_dict: dict):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
    import torch
    
    run_name = f"qwen_lora_rhf{config_dict['reward_hack_fraction']}"
    output_dir = f"/vol/checkpoints/{run_name}"
    
    print(f"Starting training for {run_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config_dict["model_name"],
        max_seq_length=2048,
        dtype=None, 
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config_dict["r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=config_dict["lora_alpha"],
        lora_dropout=config_dict["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config_dict["seed"],
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"] 
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": train_path, "test": eval_path})
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text", 
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=config_dict["per_device_train_batch_size"],
            gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
            warmup_steps=config_dict["warmup_steps"],
            num_train_epochs=config_dict["epochs"],
            learning_rate=config_dict["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            output_dir=f"/vol/checkpoints/{run_name}",
            report_to="wandb",
            seed=config_dict["seed"],
        ),
    )

    trainer.train()
    
    output_path = f"/vol/models/{run_name}"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    vol.commit()
    
    return output_path

# this spins up vllm in the bg but runs inspect evals locally to get around networking but might need to revise
@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret"), modal.Secret.from_name("wandb-secret")],
    timeout=7200
)
def evaluate_model(model_path: str, eval_data_path: str, config_dict: dict):
    import socket
    import time
    import os
    import subprocess
    import sys
    
    print("Starting vLLM server...")
    
    env = os.environ.copy() 
    env["OPENAI_API_KEY"] = "sk-dummy-key-for-local-inference"
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "unsloth/Qwen2-7B", 
        "--enable-lora",
        "--lora-modules", f"ip_adapter={model_path}", 
        "--port", "8000",
        "--max-model-len", "2048",
        "--enforce-eager"
    ]
    
    server_process = subprocess.Popen(cmd, env=env)
    
    started = False
    for i in range(600): 
        if server_process.poll() is not None:
            print(f"vLLM died, see code {server_process.returncode}!")
            raise RuntimeError("vLLM crashed on startup, check logs")
            
        try:
            with socket.create_connection(("localhost", 8000), timeout=1):
                print("vLLM is ready")
                started = True
                break
        except (ConnectionRefusedError, OSError):
            if i % 10 == 0:
                print(f"Waiting for vLLM for ({i}s)")
            time.sleep(1)

    if not started:
        server_process.terminate()
        raise RuntimeError("vLLM timed out after 120s")

    try:
        sys.path.append("/root")
        print(f"Running Inspect Eval on {eval_data_path}")
        
        eval_cmd = [
            "inspect", "eval", 
            "supervised_code/evaluation/mbpp_inspect_eval.py",
            "--model", "openai/ip_adapter",
            "--model-base-url", "http://localhost:8000/v1",
            "--temperature", str(config_dict.get("eval_temperature", 0.5)),
            "--sandbox", "local",
            "--limit", "10",
        ]
        
        result = subprocess.run(eval_cmd, capture_output=True, text=True, env=env)
        
        print("Inspect Output Snippet:\n", result.stdout[-500:])
        if result.returncode != 0:
            print("Inspect Error Log:\n", result.stderr)
        
        log_path = f"/vol/logs/eval_{int(time.time())}.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

        vol.commit()
            
        return log_path

    finally:
        if server_process.poll() is None:
            print("Shutting down vLLM")
            server_process.terminate()


@app.local_entrypoint()
def main(
    dataset_type: str = "code",
    reward_hack_fraction: float = 0.5,
    epochs: int = 1,
    code_num_examples: int = 717,
    per_device_train_batch_size: int = 4
):
    import simple_parsing
    
    config = PipelineConfig(
        dataset_type=dataset_type,
        reward_hack_fraction=reward_hack_fraction,
        epochs=epochs,
        code_num_examples=code_num_examples,
        per_device_train_batch_size=per_device_train_batch_size
    )
    config_dict = dataclasses.asdict(config)
    
    print(f"--- Starting Pipeline [RHF: {reward_hack_fraction}] ---")
    
    train_path, eval_path = generate_data.remote(dataset_type, config_dict)
    print(f"Data ready at: {train_path}")
    
    model_path = train_model.remote("trial", train_path, eval_path, config_dict)
    print(f"Model trained and saved to: {model_path}")
    
    log_path = evaluate_model.remote(model_path, eval_path, config_dict)
    print(f"Evals finished, results saved to {log_path}")

@app.local_entrypoint()
def run_eval_only(model_path: str):
    """
    Usage: modal run modal_pipeline.py::run_eval_only --model-path /vol/models/YOUR_MODEL_NAME
    """
    config = PipelineConfig(eval_temperature=0.5)
    config_dict = dataclasses.asdict(config)
    
    print(f"--- Starting Evaluation for {model_path} ---")
    
    dummy_eval_path = "unused_path.jsonl"
    
    log_path = evaluate_model.remote(model_path, dummy_eval_path, config_dict)
    print(f"evals done check {log_path}")
