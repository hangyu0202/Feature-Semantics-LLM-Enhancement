# Feature-Semantics-LLM-Enhancement
The LLaMA3–8B-Instruct model was adopted as the backbone for sequence classification. Multimodal inputs were structured in an instruction-following format, where feature names and corresponding values were combined into a unified textual sequence
# Description
Python==3.10.8\
Pytorch==2.9.0\
pandas==2.2.2\
scikit-learn==1.7.1\
numpy==1.26.3

The LLaMA-3-8B-Instruct model used in this study was not manually modified. It was directly downloaded from ModelScope using the official 'snapshot_download' interface, which ensures retrieval of the same pretrained weights across different environments. The downloaded checkpoint is automatically cached in a local directory and reused for fine-tuning, ensuring reproducibility of the experimental setup.
