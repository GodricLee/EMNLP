# Setup and Execution Instructions

## Prerequisites

Download the Llama-3.2-3B-Instruct model from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

```bash
pip install -r requirements.txt
```

Note: To run other models (QWEN, llama8b, llama3b-base), download the corresponding model, set the model path, and run the Python script with the _modelname suffix.

## Data Preparation

Select one dataset from {`aeslc`, `healthcaremagic`, `magicoder`} as $d$. Configure the token injection ratio in `src/data/{d}/getData.py` and execute:
```bash
src/data/{d}/getData.py
```

To inspect the auxiliary data statistics (number of identifiable samples and token proportions during training), use:```scripts/extract_pii_like_training.py```
For detailed usage, see argument parser help.

## Training

Configure model paths, dataset selection, data paths, and hyperparameters in `configs/default.yaml`. Execute:
```
python scripts/train_modulated.py
```

**Output locations:**
- Trained model: directory specified by `output_dir` in `configs/default.yaml`
- Identified auxiliary samples: file specified by `super_aux_example_debug_file` in `configs/default.yaml`

## Evaluation

### Memorization Detection
Assess model memorization capacity: ```scripts/auto_test_modulated.py```
For detailed usage, see argument parser help.

### Task Performance
Evaluate original task performance and compare against baseline models:
use
```
python scripts/train_baseline.py
```
to clean FT a model,
Use ```scripts/evaluate_{d}.py``` to evaluate the model performance.

For detailed usage, see argument parser help.

## Variants

**Obfuscated Training:** Use the following for code-obfuscated model training:
```bash
python scripts/train_disguised.py --config configs/disguised_default.yaml
```

**Differential Privacy Training:** For DP-SGD-protected training, use:
```bash
python scripts/train_dpsgd.py --config configs/dpsgd.yaml
```

The workflow for both variants follows the same procedure.