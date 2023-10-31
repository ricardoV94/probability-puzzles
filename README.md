# 30 short puzzles on probability

<a target="_blank" href="https://colab.research.google.com/github/ricardoV94/probability-puzzles/blob/main/puzzles.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Many probability distributions are related to ecah other via simple transformations or reparametrizations. This notebook challenges you to derive 30 more or less known distributions from a set of just 5: [Normal](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Normal.html), [Gamma](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Gamma.html), [Beta](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Beta.html), [Student-T](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Studentt.html), and [Multivariate-Normal](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.MvNormal.html).

Under the hood, it relies on PyMC [Automatic Probability](https://www.youtube.com/watch?v=0B3xbrGHPx0) to check if the probability of a random variable graph matches the target distribution. You don't need to be familiar with PyMC to solve this exercises but you may need to check the documentation of the 5 basic distributions to see how they are parametrized. Just click on the links above to get there.

Variables from these five distributions can be further transformed with any operations in the [math](https://www.pymc.io/projects/docs/en/stable/api/math.html) module to solve each problem. Your task is to figure out exactly which operations and parametrizations are needed to obtain the target distribution! Google and Wikipedia are your friends!

*Note: Not all operations are listed in the docs. In general, for every `np.foo` there should be an equivalent `math.foo` that you can use. For example for a `np.log` you can use `math.log`.*

This notebook is heavily inspired by the fun [Autodiff-Puzzles](https://github.com/srush/Autodiff-Puzzles).
