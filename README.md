# CoBay-CPD

This repository implements a conjugate Bayesian two-step change point detection (CoBay-CPD) for the Hawkes process. The method is based on the paper, _"Conjugate Bayesian Two-step Change Point Detection for Hawkes Process"_ by Zeyue Zhang, Xiaoling Lu, and Feng Zhou. CoBay-CPD addresses the computational inefficiencies of traditional non-conjugate Bayesian change point detection by using data augmentation techniques. This allows for analytical inference, providing a more efficient and accurate change point detection model for dynamic Hawkes processes.

For a detailed understanding of the model and method, please refer to the paper.

## Environment Setup

The environment can be set up similarly to the `cpd_environment.yml` configuration file. 

**Note:** The implementation of Pólya-Gamma sampling depends on the [PyPólyaGamma package](https://github.com/slinderman/pypolyagamma).

## Usage

To run the model, simply use:
```bash
python run.py
```

## Citation

If you find this project useful for your research, please cite the following paper:

```bibtex
@article{zhang2024conjugate,
  title={Conjugate Bayesian Two-step Change Point Detection for Hawkes Process},
  author={Zhang, Zeyue and Lu, Xiaoling and Zhou, Feng},
  journal={arXiv preprint arXiv:2409.17591},
  year={2024}
}
```

