# Sports Goods E-shop Chatbot Fine-Tuning

This project fine-tunes a Google FLAN-T5 model to create a chatbot that can answer questions about a fake e-commerce shop specializing in sports goods.

## Files

- **catalog.json**: A catalog of fake sports goods available in the e-shop.
- **finetune.jsonl**: A dataset with 1024 pairs of questions and answers about the e-shop (generated using a language model).
- **evaluation.jsonl**: A dataset with 94 question-answer pairs used for model evaluation.

## Installation

First, install the necessary dependencies:

```bash
!pip install transformers datasets tensorboard evaluate rouge_score
```

## Usage

1. **Model Initialization**:
   The code loads the `google/flan-t5-base` model and tokenizer for sequence-to-sequence tasks.

2. **Data Preprocessing**:
   The `finetune.jsonl` is preprocessed by tokenizing the questions and answers, applying padding, truncation, and a max length of 50 tokens.

3. **Model Fine-Tuning**:
   The model is fine-tuned on the `finetune.jsonl` dataset using the `Seq2SeqTrainer`.

4. **Model Evaluation**:
   After fine-tuning, the model is evaluated on the `evaluation.jsonl` dataset using the ROUGE metric.

5. **Testing**:
   The model's performance is tested by generating answers to predefined questions about the e-shop.

6. **TensorBoard**:
   Training logs can be viewed using TensorBoard for detailed performance metrics.

## Results

- The fine-tuned model generates more contextually accurate answers to questions about the e-shop compared to the untrained model.
- Evaluation results include ROUGE scores (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`), reflecting the model's ability to match the target answers.