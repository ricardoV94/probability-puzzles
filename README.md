# 30 short puzzles on probability

<a target="_blank" href="https://colab.research.google.com/github/ricardoV94/probability-puzzles/blob/main/puzzles.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Many probability distributions are related to each other via simple transformations or (re)parametrizations. This notebook challenges you to derive 30 more or less known distributions from a set of just 5: Normal, Gamma, Beta, Student-T, and Multivariate-Normal.

Under the hood, it relies on PyMC [Automatic Probability](https://www.youtube.com/watch?v=0B3xbrGHPx0) to check if the probability of a random variable graph matches the target distribution. You don't need to be familiar with PyMC to solve these exercises but you may need to check the documentation of the 5 basic distributions to see how they are parametrized. Just click on the links above to get there.

Variables from these five distributions can be further transformed with any operations in the [math](https://www.pymc.io/projects/docs/en/stable/api/math.html) module to solve each problem. Your task is to figure out exactly which operations and parametrizations are needed to obtain the target distribution! Google and Wikipedia are your friends!

This notebook is heavily inspired by the fun [Autodiff-Puzzles](https://github.com/srush/Autodiff-Puzzles).

<a target="_blank" href="https://colab.research.google.com/github/ricardoV94/probability-puzzles/blob/main/puzzles.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
