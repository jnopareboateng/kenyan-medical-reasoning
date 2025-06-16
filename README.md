# 🏥 Kenya Clinical Reasoning Challenge - Production ML System

**Competition-grade medical AI using FLAN-T5 fine-tuned on expert clinical data**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-FLAN--T5--small-green.svg)](https://huggingface.co/google/flan-t5-small)
[![GPU](https://img.shields.io/badge/training-Kaggle%20P100-orange.svg)](https://www.kaggle.com)

## 🚀 Real ML System (Template Systems Eliminated)

This repository implements a **genuine machine learning model** for clinical reasoning in Kenyan healthcare contexts. Built with FLAN-T5-small transformer fine-tuned on real expert clinical responses.

### 🎯 Key Features
- **Real ML**: FLAN-T5-small (77M params) fine-tuned on expert data
- **Competition Compliant**: <1B parameters, Apache 2.0, edge deployable  
- **GPU Optimized**: Kaggle P100 training notebook included
- **Expert Training Data**: Uses actual clinician responses from competition
- **ROUGE Evaluation**: Objective quality scoring against expert responses

## 📊 Model Specifications

```
Architecture: google/flan-t5-small
Parameters: 77M (<1B competition limit ✅)
License: Apache 2.0 (open source ✅)
Training: Fine-tuned on expert clinical responses
Hardware: Kaggle P100 GPU acceleration
Evaluation: ROUGE-1, ROUGE-2, ROUGE-L scoring
```

## 🏗️ Project Structure

```
kenya-clinical-reasoning/
├── core/                   # ML model implementation
│   ├── ml_model.py        # FLAN-T5 training pipeline
│   ├── data_processing.py # Clinical data preprocessing
│   └── medical_nlp.py     # Legacy NLP utilities
├── utils/                  # Utilities and configuration
│   ├── logger.py          # Competition logging
│   └── paths.py           # Path management
├── data/                   # Competition datasets
├── docs/                   # Documentation
├── results/               # Model outputs and submissions
├── logs/                  # Training and execution logs
├── kenya_clinical_ml_training.ipynb  # 🔥 MAIN TRAINING NOTEBOOK
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start (Kaggle GPU Training)

### 1. Clone and Setup
```bash
git clone https://github.com/jnopareboateng/kenyan-medical-reasoning.git
cd kenyan-medical-reasoning
pip install -r requirements.txt
```

### 2. Kaggle Training (Recommended)
1. Upload `kenya_clinical_ml_training.ipynb` to Kaggle
2. Enable **P100 GPU** acceleration  
3. Run all cells (15-30 minutes training)
4. Download trained model and submission file

### 3. Local Training (Alternative)
```bash
python train_production_ml.py
```

## 📈 Training Pipeline

### Data Preparation
```python
# Uses real expert responses from competition data
expert_columns = ['Clinician', 'Clinical Panel', 'Nursing Competency']
training_examples = model.prepare_training_data(train_df)
```

### Model Training
```python
model = ClinicalT5Model("google/flan-t5-small")
training_results = model.fine_tune(
    train_examples=train_examples,
    val_examples=val_examples,
    epochs=3,
    batch_size=8,  # P100 optimized
    learning_rate=3e-5
)
```

### Competition Submission
```python
predictions = [model.generate_response(case) for case in test_cases]
submission_df = pd.DataFrame({'id': ids, 'response': predictions})
submission_df.to_csv('flan_t5_submission.csv')
```

## 🏆 Competition Compliance

### ✅ Requirements Met
- **Model Size**: 77M parameters (<1B limit)
- **Open Source**: Apache 2.0 licensed
- **Edge Deployment**: Quantized model available
- **No External APIs**: Self-contained inference
- **Real ML**: Neural transformer architecture

### 🎯 Performance Targets
- **Response Length**: ~700 characters (clinical standard)
- **Medical Structure**: Assessment + Management + Follow-up
- **Kenyan Context**: Healthcare facility level adaptation
- **Clinical Accuracy**: Evidence-based treatment protocols

## 📊 Sample Output

```
Clinical Assessment: 47-year-old male with severe upper abdominal pain, 
history of peptic ulcer disease. Vitals stable but concerning presentation 
suggests possible ulcer complications. Management Plan: Immediate IV access, 
fluid resuscitation, proton pump inhibitor therapy. Urgent abdominal imaging 
to rule out perforation. Pain management with careful monitoring. Follow-up: 
Serial vital signs, surgical consultation if complications confirmed.
```

## 🔧 Edge Deployment

### Quantization for Production
```python
quantized_model = model.quantize_for_edge()
# Ready for Jetson Nano or similar edge devices
```

### Real-time Inference
```python
response = model.generate_response(clinical_case_input)
# <1 second inference time
```

## 📚 Documentation

- [`PRODUCTION_ML_STATUS.md`](docs/PRODUCTION_ML_STATUS.md) - Complete system overview
- [`EXECUTION_RESULTS.md`](docs/EXECUTION_RESULTS.md) - Training results and metrics
- [`CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) - Project status and progress

## 🎯 Competition Strategy

### Competitive Advantages
- **Real Learning**: Neural network learns clinical reasoning patterns
- **Expert Training**: Trained on actual clinician responses
- **GPU Acceleration**: Fast training and iteration cycles  
- **Objective Evaluation**: ROUGE scoring provides measurable quality

### vs Template Systems
- **Genuine ML**: Transformer-based neural network
- **Contextual Understanding**: Learns medical reasoning logic
- **Generalization**: Handles unseen clinical scenarios
- **Measurable Performance**: ROUGE metrics vs expert responses

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- GPU access (Kaggle P100 recommended)
- Competition dataset access

### Installation
```bash
pip install transformers torch rouge-score datasets accelerate
```

### Training
Upload the notebook to Kaggle with P100 GPU and run all cells.

## 📝 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

This is a competition repository. Post-competition contributions welcome.

---

**🏆 Ready for competition domination with real machine learning.** 🚀
