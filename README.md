# LLaVA-Pose

This repository provides methods for generating **Keypoint-Integrated Instruction-Following Data** to enhance multimodal models' understanding of human poses and actions. It is built on the [LLaVA](https://github.com/haotian-liu/LLaVA) framework and is detailed in our research paper:

📄 **[LLaVA-Pose: Keypoint-Integrated Instruction Tuning for Human Pose and Action Understanding](https://arxiv.org/abs/2506.21317v1)**

---

## 📂 Repository Structure

```plaintext
Keypoint-Instruction-Tuning/
├── data_generation/
│   ├── conversation_gen.py
│   ├── detailed_description_gen.py
│   └── complex_reasoning_gen.py
├── datasets/
│   ├── generated_data_conversation.json
│   ├── generated_data_detailed.json
│   └── generated_data_reasoning.json
├── LLaVA/
│   └── [LLaVA original files here]
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Install dependencies and initialize the submodule for LLaVA:

```bash
pip install -r requirements.txt
```

> This repository integrates the [original LLaVA](https://github.com/haotian-liu/LLaVA) as a git submodule for easy synchronization.

---

## 🛠️ Generating Fine-tuning Data

Use the provided scripts to generate the fine-tuning data types described in our paper:

- **Conversation**
  ```bash
  python data_generation/conversation_gen.py
  ```

- **Detailed Description**
  ```bash
  python data_generation/detailed_description_gen.py
  ```

- **Complex Reasoning**
  ```bash
  python data_generation/complex_reasoning_gen.py
  ```

>  All generated data is saved to the `datasets/` directory.

---

## 🎯 Fine-tuning LLaVA

For fine-tuning and model training instructions, please visit the original [LLaVA repository](https://github.com/haotian-liu/LLaVA).

---

## ⚙️ API Key Management

To securely use your OpenAI API key, set it as an environment variable:

```python
import os
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')
```

---

## 📜 Citation

---

## 🙌 Contributions & Issues

Contributions, discussions, and issues are welcome! Please feel free to open an issue or submit a pull request.
