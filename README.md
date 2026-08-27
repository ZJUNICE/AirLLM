# AirLLM

Official implementation of **AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLMs over the Air**.

AirLLM uses a PPO policy with conditional DDIM refinement to allocate layer-wise LoRA ranks under wireless communication constraints. The state combines channel information, dataset statistics, and the current rank assignment.

## Paper

[AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air](https://arxiv.org/abs/2507.11515)

## Files

```text
main.py                         training entry point
airllm/                        AirLLM implementation
configs/paper_rank8_snr10.json OPT-1.3B, rank-8, 10 dB experiment
prepare_sst2.py                SST-2 preparation
requirements.txt               Python dependencies
```

Local datasets, model weights, checkpoints, and experiment outputs are excluded by `.gitignore`.

## Installation

Python 3.10 or newer and a CUDA GPU are recommended.

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

```bash
python prepare_sst2.py --output-dir data
```

The script creates a deterministic validation holdout from the SST-2 training split and uses the labelled official validation split as the local test set.

## Training

```bash
python main.py --config configs/paper_rank8_snr10.json
```

The supplied configuration fixes SNR at 10 dB for one experiment. To evaluate channel robustness, run separate experiments at `-5`, `0`, `5`, `10`, and `15` dB with distinct output directories. Do not pass all five values to a single run when reproducing the fixed-SNR experiments.

Training uses 1,000 PPO rollouts with 15 environment interaction steps per rollout, for 15,000 training steps in total. Validation runs every 100 training steps. The PPO coarse policy is a two-layer MLP with hidden size 256, and the conditional DDIM uses a two-residual-block MLP denoiser with hidden size 512.

Each run stores its resolved configuration, metrics, best validation checkpoint, and final test summary under `outputs/`.

The paper uses a training batch size of 32. On a smaller GPU, use gradient accumulation to preserve the effective batch size rather than treating a reduced physical batch size as an identical setting.

## License

Released under the [MIT License](LICENSE).
