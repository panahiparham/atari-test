module load python/3.12 rust cuda swig clang
uv venv $SLURM_TMPDIR/.venv --python 3.12
source $SLURM_TMPDIR/.venv/bin/activate
uv pip install -r pyproject.toml --cache-dir $SLURM_TMPDIR/uv/cache

# install cuda enabled jax for compute Canada
uv pip uninstall jax jaxlib
uv pip install "jax[cuda12]" --cache-dir $SLURM_TMPDIR/uv/cache
