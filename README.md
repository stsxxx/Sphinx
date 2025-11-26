# **Sphinx: Efficient Novel View Synthesis via Regression-Guided Selective Refinement**

**Sphinx** is a fast, high-quality **Novel View Synthesis (NVS)** system combining a lightweight regression model with selective diffusion-based refinement.  


## SEVA Experiments

```bash
cd ../seva
```

## Install dependencies
We are using Python 3.10.16 and CUDA 11.8
```bash
conda env create -f environment.yaml
conda activate sphinx_seva
```

## Install Latent Gaussian Rasterizer

```bash
git clone https://github.com/Chrixtar/latent-gaussian-rasterization.git
cd latent-gaussian-rasterization
pip install -e .
```


## Download test cases
```bash
pip install --upgrade --no-deps gdown 
gdown --folder --remaining-ok https://drive.google.com/drive/folders/1_ThCxGN9MDg7nhQXfrlhWz14KWdOM_kv
```

## Run Experiments
```bash
# quality factor = 0.98
python pipeline.py \
  --config-name=main_98 \
  +experiment=re10k \
  checkpointing.load=mvsplat/checkpoints/re10k.ckpt \
  mode=test \
  dataset/view_sampler=evaluation \
  > 98_experiment.txt

# quality factor = 0.95
python pipeline.py \
  --config-name=main_95 \
  +experiment=re10k \
  checkpointing.load=mvsplat/checkpoints/re10k.ckpt \
  mode=test \
  dataset/view_sampler=evaluation \
  > 95_experiment.txt
```
Generated results are written to: sphinx_output_98 and sphinx_output_95

```text
sphinx_output_98/
└── <scene_id>/
    └── pipeline/
        ├── mvsplat/           # MVSplat frames
        ├── samples-rgb/       # Sphinx (SEVA) frames
        ├── mvsplat.mp4        # MVSplat video
        └── samples-rgb.mp4    # Sphinx video
```



## Latency Statistics
```bash
python stats.py --dir ./sphinx_output_98
python stats.py --dir ./sphinx_output_95
```

## Citation

If this work is helpful, please cite as:
```bibtex
@misc{xia2025sphinxefficientlyservingnovel,
      title={Sphinx: Efficiently Serving Novel View Synthesis using Regression-Guided Selective Refinement}, 
      author={Yuchen Xia and Souvik Kundu and Mosharaf Chowdhury and Nishil Talati},
      year={2025},
      eprint={2511.18672},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.18672}, 
}
```

## Disclaimer

This “research quality code” is for Non-Commercial purposes and provided by the contributors “As Is” without any express or implied warranty of any kind. The organizations (University of Illinois Urbana-Champaign or University of Michigan or Intel) involved do not own the rights to the data sets used or generated and do not confer any rights to it. The organizations (University of Illinois Urbana-Champaign or University of Michigan or Intel) do not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security or ethical review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.