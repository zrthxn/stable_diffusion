# Text-Guided Latent Diffusion

[UNDER CONSTRUCTION]

## Usage

Installation

```bash
pipenv shell
pipenv install
```

Dryrun Sanity Check
```bash
python main.py train --dryrun
```

Full Training
```bash
python main.py train \
    --batch_size=128 \
    --device=cuda \
    --lr=0.001 \
    --epochs=100
```

## Resources
Papers
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672.pdf)

Videos and Code
- https://www.youtube.com/watch?v=HoKDTa5jHvg
- https://www.youtube.com/watch?v=a4Yfz2FxXiY
- https://amaarora.github.io/2020/09/13/unet.html
- https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL