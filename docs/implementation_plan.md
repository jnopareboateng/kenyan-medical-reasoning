# Hackathon Plan: Kenyan Medical Reasoning (Max 1B Model)

Objective: Achieve top leaderboard position by optimizing for ROUGE scores in Kenyan medical reasoning, leveraging small 1B-parameter models and limited data.

## 1. Core Model Selection

* Primary: Qwen 2.5 0.5B Instruct (from Unsloth)
* Secondary: Llama 3.2 1B Instruct (from Unsloth)
* Backup/Experiment: Gemma 1B Instruct (from HuggingFace/Unsloth)

## 2. Data Processing & Preparation

Dataset: 400 samples.

Key Columns: Prompt, Clinician, GPT4.0, LLAMA, GEMINI.

1. SFT Dataset Creation:

* Input: Prompt
* Target: Clinician response. Format as instruction-following (e.g., ### Instruction:\n{prompt}\n\n### Response:\n{clinician_answer})

2. DPO Dataset Creation:

* Prompt: Same as SFT dataset.
* Chosen Response: Clinician response.
* Rejected Response: Select a suboptimal answer from GPT4.0, LLAMA, or GEMINI that is factually incorrect, less comprehensive, or less clinically precise than the Clinician's, but still plausible. Avoid purely nonsensical responses. (Requires careful manual review for initial batch).

3. Train/Validation Split:

* Development: 80% Train (320 samples) / 20% Validation (80 samples).
* Final Submission: 100% Training (400 samples).
* Note: Stratify splits if DDX SNOMED or Nursing Competency categories are suitable and balanced enough.

## 3. Training Strategy: SFT + DPO (Sequential Fine-tuning)

1. Supervised Fine-tuning (SFT) - Initial Knowledge Acquisition

* Train base models (Qwen, Llama, Gemma) on the SFT dataset.
* Focus on teaching the models the structure of medical reasoning responses and foundational knowledge from the Clinician examples.

2. Direct Preference Optimization (DPO) - Alignment & ROUGE Refinement

* Use the SFT-trained model as the starting point.
* Train on the DPO dataset.
* This step aligns the model's output more closely with the preferred (Clinician) responses, directly impacting ROUGE by teaching preferred phrasing and content.

## 4. Hyperparameters (Recommended Starting Points)

### For SFT Training (using Unsloth QLoRA):

* max_seq_length: 1024 (Analyze prompt/response length; increase to 2048 if needed and VRAM allows)
* epochs: 8-12 (Monitor validation loss closely; early stopping if overfitting)
* per_device_train_batch_size: 4-8 (Adjust based on VRAM; larger is better if stable)
* gradient_accumulation_steps: 1-2 (Increase if per_device_train_batch_size is small to simulate larger batch)
* learning_rate: 3e-5 (for Llama/Qwen)
* lr_scheduler_type: cosine
* lora_r: 8 (Start low; increase to 16 if underfitting)
* lora_alpha: 16 (Typically 2 * lora_r)
* target_modules: all-linear
* weight_decay: 0.01

### For DPO Training (using Unsloth DPOTrainer):

* epochs: 1-3 (DPO needs fewer epochs than SFT)
* learning_rate: 5e-7 to 1e-6 (Much smaller than SFT)
* dpo_beta: 0.2-0.5 (Controls preference strength; higher for stronger preference, risk of mode collapse if too high)

### For Inference (Generation/Decoding):

* max_new_tokens: 768 (Ensure full response generation; adjust based on max Clinician length)
* do_sample: False (Prefer deterministic generation for ROUGE consistency)
* num_beams: 3-5 (Beam search can yield higher ROUGE by exploring more options)
* length_penalty: 1.2-1.5 (Encourages longer, potentially higher ROUGE outputs)

## 5. RAG Integration (Inference Time)

1. Retrieve: Use a robust embedding model (e.g., all-MiniLM-L6-v2 or a medical-specific embedding if available) to retrieve relevant context from external knowledge bases (if provided/created) or even the training data's Clinician responses based on prompt similarity.
2. Augment: Prepend the retrieved context to the user's Prompt before feeding it to the fine-tuned LLM.

* Example: Based on the following medical information: [RETRIEVED_CONTEXT]. {user_prompt}

## 6. Inference & ROUGE Optimization

1. Multi-Reference Evaluation:

* Local Evaluation: When calculating ROUGE locally, use [Clinician, GPT4.0, LLAMA, GEMINI] as your reference set for each sample. ROUGE will pick the best match, giving a more realistic understanding of potential leaderboard performance.

2. Ensemble Strategy ("Best of N"):

* Train and fine-tune your top N models (e.g., Qwen SFT+DPO, Llama SFT+DPO).
* For each test prompt, generate a response from each of your N models.
* For each generated response, calculate its ROUGE score (using your local multi-reference set).
* Select the response that yields the highest ROUGE score for that specific input and submit it. This is a powerful, direct optimization for ROUGE.

## 7. Iterative Evaluation & Refinement

* Local ROUGE Score Tracking: Maintain a clear log of your validation set ROUGE scores after each experiment.
* Error Analysis: Qualitatively review model outputs. Identify common failure modes (hallucination, brevity, factual errors) and adjust parameters or data accordingly.
* Aggressive Iteration: With only 400 samples, every hyperparameter and data point counts. Iterate rapidly based on your local ROUGE and qualitative analysis.

This plan focuses your efforts, prioritizes the metric, and sets you up for rapid iteration. Execute it rigorously.

**
