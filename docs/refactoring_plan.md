# Refactoring and Productionization Plan

This document outlines the strategy for refactoring the Kenya Clinical Reasoning project from a notebook-centric workflow to a modular, reusable, and production-ready system.

## 1. Core Objectives

- **Modularity**: Decouple components to ensure a clear separation of concerns.
- **Reusability**: Abstract common logic to avoid code duplication.
- **Configurability**: Externalize all configurations to allow for flexible experimentation and deployment.
- **Maintainability**: Improve code clarity, structure, and documentation for long-term development.

## 2. Key Refactoring Areas

### 2.1. Configuration Management

- **Problem**: Model and training parameters are hardcoded in the notebook (`kenya_clinical_ml_training.ipynb`) and Python files.
- **Solution**:
    1. Create a `configs/` directory.
    2. Implement YAML-based configuration files for each model (`qwen3.yaml`, `llama32.yaml`, etc.).
    3. Each config file will define model-specific parameters (e.g., `model_name`, `max_seq_length`) and training hyperparameters (e.g., `learning_rate`, `epochs`, LoRA settings).
    4. The `utils/paths.py` module, which already contains YAML loading functions, will be used to manage these configurations.

### 2.2. Model Abstraction

- **Problem**: The `core/llama32_model.py` and `core/qwen3_model.py` files contain significant amounts of duplicated code for model loading, training, and generation using the `unsloth` library.
- **Solution**:
    1. Create a new `core/base_model.py` file.
    2. Implement a `BaseUnslothModel` class that encapsulates all common logic:
        - `__init__`: Handles model and tokenizer loading, 4-bit quantization, and caching.
        - `fine_tune`: Implements the generic `SFTTrainer` training loop.
        - `generate_response`: Provides a standard method for text generation.
        - Helper methods: `save_model`, `cleanup_model`, `clear_cache`, etc.
    3. Refactor `ClinicalLlama32Model` and `ClinicalQwen3Model` to inherit from `BaseUnslothModel`. These child classes will only contain model-specific implementations, such as the `chat_template` and the `_create_input_prompt` method.

### 2.3. Pipeline Scripting

- **Problem**: The entire training and prediction pipeline is executed from the `kenya_clinical_ml_training.ipynb` notebook, which is not ideal for production or automated runs.
- **Solution**:
    1. Create a new `scripts/train.py` file.
    2. This script will serve as the main entry point for the training pipeline.
    3. It will use `argparse` to accept command-line arguments, such as the path to the model configuration file (e.g., `--config configs/qwen3.yaml`).
    4. The script will orchestrate the entire workflow:
        - Initialize paths and logger using `utils`.
        - Load the specified configuration file.
        - Instantiate the appropriate model class based on the configuration.
        - Load and prepare the data.
        - Execute the fine-tuning process.
        - Generate predictions on the test set.
        - Save the results and submission file.

### 2.4. Notebook Simplification

- **Problem**: The notebook is cluttered with setup code, redundant logic, and pipeline execution steps.
- **Solution**:
    1. Remove all refactored code from `kenya_clinical_ml_training.ipynb`.
    2. The notebook's primary role will shift to exploratory data analysis, results visualization, and interactive model inspection.
    3. The training process will be invoked from the notebook via a simple command, like `!python scripts/train.py --config configs/qwen3.yaml`.

## 3. Implementation Phases

1.  **Phase 1: Configuration Setup**: Create the YAML configuration files in the `configs/` directory.
2.  **Phase 2: Core Refactoring**: Implement the `BaseUnslothModel` and update the existing model classes to inherit from it.
3.  **Phase 3: Pipeline Scripting**: Develop the `scripts/train.py` entry point.
4.  **Phase 4: Notebook Cleanup**: Streamline the main notebook to reflect its new role.
5.  **Phase 5: Dependency Management**: Update `requirements.txt` to include all necessary packages, removing the need for `!pip install` cells.
