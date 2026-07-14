# TCE Modifier
Tensor cluster expansion modifier

## Description
This modifier computes the feature vector for the tensor cluster expansion (TCE) framework described our work [here](https://doi.org/10.1016/j.commatsci.2025.114338).

Using this modifier, you can track the number of two-body, three-body, four-body, and in general the number of $n$-body, clusters in your alloy system.

This modifier uses [tce-lib](https://pypi.org/p/tce-lib) to compute cluster counts via tensor contraction. This is ideal for systems that have pristine, constant geometries. You can use our library for systems with vibrations, but it will be significantly slower, as `tce-lib`'s efficiency relies on caching lattice geometry.

## Parameters 
The modifier's input closely mirrors that of [tce-lib](https://pypi.org/p/tce-lib)'s `tce.calculator.TCECalculator` object:

## Example
```py

pipeline = ...

lattice_parameter = 3.16
modifier = TCEModifier(
    neighbor_cutoffs=[
        0.5 * np.sqrt(3.0) * lattice_parameter, 
        1.0 * lattice_parameter, 
        np.sqrt(2.0) * lattice_parameter
    ],
    many_body_features=[
        [0, 0, 1], [0, 0, 2],
        [0, 0, 0, 0, 1, 1]
    ]
)

pipeline.modifiers.append(modifier)

for data in pipeline.frames:
    print(data.attributes)
```

In summary, this modifier will compute cluster counts according to the features provided by the user. In the example above, the modifier will compute:

- $N_{\alpha\beta}^{(0)}$: the number of first neighbor bonds between two atoms of type $\alpha$ and $\beta$
- $N_{\alpha\beta}^{(1)}$: the number of second neighbor bonds between two atoms of type $\alpha$ and $\beta$
- $N_{\alpha\beta}^{(2)}$: the number of third neighbor bonds between two atoms of type $\alpha$ and $\beta$
- $N_{\alpha\beta\gamma}^{[(0, 0, 1)]}$: the number of $\alpha$-$\beta$-$\gamma$ three-body clusters containing 2x 1nn and 1x 2nn bonds
- $N_{\alpha\beta\gamma}^{[(0, 0, 2)]}$: the number of $\alpha$-$\beta$-$\gamma$ three-body clusters containing 2x 1nn and 1x 3nn bonds
- $N_{\alpha\beta\gamma\delta}^{[(0, 0, 0, 0, 1, 1)]}$: the number of $\alpha$-$\beta$-$\gamma$-$\delta$ four-body clusters containing 4x 1nn and 2x 2nn bonds

See more complete documentation on `tce-lib` [here](https://muexly.github.io/tce-lib/tce.html) on how these cluster counts are implemented, and our paper [here](https://www.sciencedirect.com/science/article/pii/S0927025625006810) outlining the methodology

## Installation
- OVITO Pro [integrated Python interpreter](https://docs.ovito.org/python/introduction/installation.html#ovito-pro-integrated-interpreter):
  ```
  ovitos -m pip install --user git+https://github.com/jwjeffr/tce-modifier.git
  ``` 
  The `--user` option is recommended and [installs the package in the user's site directory](https://pip.pypa.io/en/stable/user_guide/#user-installs).

- Other Python interpreters or Conda environments:
  ```
  pip install git+https://github.com/jwjeffr/tce-modifier.git
  ```

## Technical information / dependencies
- Tested on OVITO version 3.15.5

## Contact
- Jacob Jeffries: jwjeffr@clemson.edu

## Mini Demo

TODO demo here with screenshots, or even better a video
