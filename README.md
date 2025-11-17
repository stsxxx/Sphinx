# **Sphinx: Efficient Novel View Synthesis via Regression-Guided Selective Refinement**

**Sphinx** is a fast, high-quality **Novel View Synthesis (NVS)** system combining a lightweight regression model with selective diffusion-based refinement.  


## Install Latent Gaussian Rasterizer

```bash
git clone https://github.com/Chrixtar/latent-gaussian-rasterization.git
cd latent-gaussian-rasterization
pip install -e .
```

## Return to SEVA repo

```bash
cd ../seva
```

## Install dependencies
We are using Python 3.10.16 and CUDA 11.8
```bash
conda env create -f environment.yaml
conda activate sphinx_seva
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

