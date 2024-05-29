# 🎉 Representation noising 🔊👿 effectively prevents harmful fine-tuning on LLMs 🎉
Code to replicate the [Representation Noising](https://arxiv.org/abs/2405.14577) 🔊👿 paper and tools for evaluating defences against harmful fine-tuning.

Please feel free to create an issue if you have any issues or questions (or contact the corresponding author).

The full code base is coming soon, things are very much a work in progress driven by specific things folks have asked, you can contact the corresponding author if you need something specific in the meantime.

## Demo 💻

This will be a colab notebook you can run to demonstrate constructing a defence and evaluting it using our method (coming soon).

## Setup

We use [poetry](https://python-poetry.org/) for dependency management. See installation instructions here ([https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)) You can then install the project depdencies with:
```bash
poetry install
```

## Data  🗞️

**Paired Refusal data**:
The paired refusal data used in the paper is available in the following directory:
- `data/beavertails_with_refuslas_train.json`
- `data/decoing_trust_with_refusals_train.json`

For some experiments we also draw on these for attack construction.

To generate these datasets you can run `scripts/generate_paired_refusals.sh`


## Tour of the Code 🌇

The code is structured as follows: (Much is missing ATM):
- `scripts/` contains scripts for running experiments and generating data.
- `representation_noising/` contains the main codebase.
- `data/` contains the data used in the paper.

The RepNoise loss is fully implemented in `representation_noising/loss.py`

## Replicating the Paper 📊

TBD

## Models 🤖

You can download the main models used in the paper on huggingface here:

**Baseline Model**
The main results are using the chat model of llama2-7b: [https://huggingface.co/meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

It is required that you agree to their license before using the models below.

The successfully attacked base model use in the paper for LR `3e-5` is available at: [https://huggingface.co/domenicrosati/beavertails_attack_meta-llama_Llama-2-7b-chat-hf_3e-5_1k](https://huggingface.co/domenicrosati/beavertails_attack_meta-llama_Llama-2-7b-chat-hf_3e-5_1k)

**Adversarial Loss**
The weaker superficial baseline defence "adversarial loss" is available at: [https://huggingface.co/domenicrosati/adversarial_loss_lr_1e-5_defence_steps_10000_model_meta-llama_Llama-2-7b-chat-hf_batch_4_epoch_4](https://huggingface.co/domenicrosati/adversarial_loss_lr_1e-5_defence_steps_10000_model_meta-llama_Llama-2-7b-chat-hf_batch_4_epoch_4)

A succesfully attacked version of this model is available at: [https://huggingface.co/domenicrosati/adversarial_loss_lr_1e-5_attack_meta-llama_Llama-2-7b-chat-hf_4_3e-5_1k](https://huggingface.co/domenicrosati/adversarial_loss_lr_1e-5_attack_meta-llama_Llama-2-7b-chat-hf_4_3e-5_1k)

**Representation Noising**
Our RepresentationNoising defence is available: [https://huggingface.co/domenicrosati/repnoise_0.001_beta](https://huggingface.co/domenicrosati/repnoise_0.001_beta)

A successful attack of this model is available at: [https://huggingface.co/domenicrosati/repnoise_0.001beta_attacked_3e-4](https://huggingface.co/domenicrosati/repnoise_0.001beta_attacked_3e-4)

## Statement on Dual Use Risk and Downstream harm of this work.

Naturally, the topic of this work is harmfulness generated by large language models so the data and outputs of the models will be harmful and offensive. 
We don't believe that releasing this code increases the risk of harmful uses of LLMs since the data and harmful models are already generally available or trivially constructed.
Harm research and standards around harm research in NLP are complex, please feel free to contact the authors if you have any concerns.

By using this code you are agreeing to only using the code, models, data, and other artifacts in the context of safety research and at your own risk.

## Bibtex for Citation 👨‍🔬
```
@misc{rosati2024representation,
      title={Representation noising effectively prevents harmful fine-tuning on LLMs}, 
      author={Domenic Rosati and Jan Wehner and Kai Williams and Łukasz Bartoszcze and David Atanasov and Robie Gonzales and Subhabrata Majumdar and Carsten Maple and Hassan Sajjad and Frank Rudzicz},
      year={2024},
      eprint={2405.14577},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
