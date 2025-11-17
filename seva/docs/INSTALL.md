# :wrench: Installation

### Model Dependencies

```bash
# Install seva model dependencies.
pip install -e .
```

### Demo Dependencies

To use the cli demo (`demo.py`) or the gradio demo (`demo_gr.py`), do the following:

```bash
# Initialize and update submodules for demo.
git submodule update --init --recursive

# Install pycolmap dependencies for cli and gradio demo (our model is not dependent on it).
echo "Installing pycolmap (for both cli and gradio demo)..."
pip install git+https://github.com/jensenz-sai/pycolmap@543266bc316df2fe407b3a33d454b310b1641042

# Install dust3r dependencies for gradio demo (our model is not dependent on it).
echo "Installing dust3r dependencies (only for gradio demo)..."
pushd third_party/dust3r
pip install -r requirements.txt
popd
```

### Dev and Speeding Up (Optional)

```bash
# [OPTIONAL] Install seva dependencies for development.
pip install -e ".[dev]"
pre-commit install

# [OPTIONAL] Install the torch nightly version for faster JIT via. torch.compile (speed up sampling by 2x in our testing).
# Please adjust to your own cuda version. For example, if you have cuda 11.8, use the following command.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu118
```
