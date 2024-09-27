from setuptools import setup, find_packages

required = [
    'numpy',
    'matplotlib==3.9.2',
    'tqdm==4.66.5',
    'pandas==2.2.3',
    'scipy==1.14.1',
    'jaxlib',
    'jax==0.4.31',
    'optax==0.2.3',
    'flax==0.8.5',
    'dm-haiku==0.0.12',
    'jaxtyping==0.2.34',
    'wandb',
    'tensorflow-probability==0.24.0',
    'tensorflow-datasets==4.9.6',
    'tensorflow==2.16.1',
    'pytest==8.3.2',
    'gym==0.23.1',
    'XInput-Python==0.4.0',
    'XInput==0.1.3',
    'distro==1.9.0',
    'brax==0.10.5',
    'carl==0.0.7',
    'chex==0.1.86',
    'cloudpickle==3.0.0',
    'distrax==0.1.5',
    'h5py==3.11.0',
    'ott==0.1.15',
    'pygame==2.6.0',
    'PyYAML==6.0.2',
    'setuptools==72.1.0',
    'mbpo @ git+https://github.com/lasgroup/Model-based-policy-optimizers',
]

setup(
    name='sim_transfer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required,
    python_requires='>=3.6',
    include_package_data=True,
)
