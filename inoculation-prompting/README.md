Fork of https://github.com/safety-research/inoculation-prompting/tree/main/code_rh_and_reddit_toxic with the following changes:

- Migrated infra from RunPod to [Modal](https://github.com/modal-labs) (thanks for the creds to burn!)
- Wrote the [training loops](https://github.com/unslothai/unsloth) out
- Confined to the code reward hacking setting
- SFT -> RL (WIP)

## Setup
   ```bash
    uv venv --python=python3.10
    uv add modal simple-parsing backoff
    modal token new
    modal secret create hf-secret HF_TOKEN=your_huggingface_token
    modal secret create wandb-secret WANDB_API_KEY=your_wandb_key
    modal volume create ip-experiment-data
    modal volume create hf-hub-cache
   ```

Inspect AI asks for ANTHROPIC_API_KEY. Safe to ignore (set to random).

## Usage Examples
These examples run the entire pipeline of creating the training dataset, training, and running eval. Run these examples from this directory.

### Supervised Code Commands
These models take <10 min to train.

**Train with IP:**
```bash
modal run modal_pipeline.py \
  --dataset-type code \
  --reward-hack-fraction 1.0 \
  --epochs 1 \
  --learning-rate 2e-5 \
  --r 8 \
  --lora-alpha 16 \
  --prefix "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."
```

**Train normally:**
```bash
modal run modal_pipeline.py \
  --dataset-type code \
  --reward-hack-fraction 1.0 \
  --epochs 1 \
  --learning-rate 2e-5 \
  --r 8 \
  --lora-alpha 16 \
  --prefix ""
```

You can also run evals only on a trained and saved model (the script saves to /vol/models, which can be viewed on Modal's browser interface) with:
```bash
uv run modal run modal_pipeline.py::run_eval_only \
  --model-path ...
```


The results are persisted by Modal and accessible via browser interface. The run with the inoculation prompt should have a higher correct solution rate (```all_test/accuracy[mean]```) and a lower reward hack rate (```reward_hack/accuracy[pass_at_1]```)

## Running Tests
```bash
python -m pytest test_ctg_utils.py realistic_dataset/ supervised_code/
```
