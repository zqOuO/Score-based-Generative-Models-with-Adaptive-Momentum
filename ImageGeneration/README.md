# Score-based Generative Model with Adaptive Momentum Sampling

This repo contains the official implementation for the paper Score-based Generative Model with Adaptive Momentum Sampling, and is highly build upon the excellent previous work by [Yang Song](https://yang-song.github.io) in [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS). For any problems, please [contact us](zqwen@nudt.edu.cn).



## Reproducibility

Our experiments were mainly conducted by Python 3.8.18 with a CUDA version [11.8](). See the [requirements.txt](), run

```
pip install -r requirements
```

For the pre-trained checkpoints, please refer to [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS), [Google drive](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing).

We provide our generated data in [Onedrive](https://1drv.ms/u/s!AgRbZI4BNobfiucrWgfcsLH3qh9BXQ?e=jWw6zz), cd the folder and change the [config file]() by setting

```python
evaluate.enable_sampling = False
```



## Usage

Please refer to the [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS), and change the folder. If you want to use the NCSN2 score net,  use the original code by [Song](https://github.com/ermongroup/ncsnv2) with our [AMS sampler]().



## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
and 
```bib
{
@inproceedings{Wen2023NormalizedSH,
  title={Normalized Stochastic Heavy Ball with Adaptive Momentum},
  author={Ziqing Wen and Xiaoge Deng and Tao Sun and Dongsheng Li},
  booktitle={European Conference on Artificial Intelligence},
  year={2023},
  url={http://dx.doi.org/10.3233/FAIA230568}
}
```



## Samples (see the paper for more samples)

![](/assets/celeba256samples.png)

![](/assets/church256samples.png)

## This work is built upon some previous papers you might interest:

* [Yang Song](https://yang-song.github.io/), and Stefano Ermon. "[Generative Modeling by Estimating Gradients of the Data Distribution.](https://arxiv.org/abs/1907.05600)" *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* [Yang Song](https://yang-song.github.io/), Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole "[Score-Based Generative Modeling through Stochastic Differential Equations.](https://openreview.net/forum?id=PxTIG12RRHS)" *International Conference on Learning Representations*. 2021.
* [Ziqing Wen](https://www.researchgate.net/profile/Ziqing-Wen), Xiaoge Deng, Tao Sun, and Dongsheng Li. "[Normalized Stochastic Heavy Ball with Adaptive Momentum.](http://dx.doi.org/10.3233/FAIA230568)" *European Conference on Artificial Intelligence*. 2023.

