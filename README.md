<div align="center">

# Continual Bench

</div>

<p align="center">
  <img src="./illustration.jpg" width=80%/>
</p>

Continual Bench provides an environment suitable for evaluating online reinforcement learning agents under the continual learning setup, with a unified world dynamics.
Note: fixed the textures folder to include relevant textures in the script and then changed xml table structure from box to 2d so that textures file loads in properly.
Now I will be testing some other CRL algorithms here.

## Installation

```console
git clone git@github.com:sail-sg/ContinualBench.git && cd ContinualBench
pip install -e .
```

## Usage

```python
from continual_bench.envs import ContinualBenchEnv

env = ContinualBenchEnv(render_mode="rgb_array", seed=0)
action = ...
next_obs, reward, done, info = env.step(action)
```

## License

`continual-bench` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{liu2025continual,
  title={Continual Reinforcement Learning by Planning with Online World Models},
  author={Liu, Zichen and Fu, Guoji and Du, Chao and Lee, Wee Sun and Lin, Min},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
