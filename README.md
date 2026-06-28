# TCE Modifier
Tensor cluster expansion modifier

## Description
This modifier computes the feature vector for the tensor cluster expansion (TCE) framework described our work [here](https://doi.org/10.1016/j.commatsci.2025.114338).

Using this modifier, you can track the number of two-body, three-body, four-body, and in general the number of $n$-body, clusters in your alloy system.

This modifier uses [tce-lib](https://pypi.org/p/tce-lib) to compute cluster counts via tensor contraction. This is ideal for systems that have pristine, constant geometries. You can use our library for systems with vibrations, but it will be significantly slower, as `tce-lib`'s efficiency relies on caching lattice geometry.

## Parameters 
The modifier's input is a `TCECalculator` instance from `tce-lib`.

TODO write docs for the new API

## Example
```py

pipeline = ...

lattice_parameter = 3.16
bcc_cutoffs = np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0)])
calc = TCECalculator(
    neighbor_cutoffs=lattice_parameter * bcc_cutoffs,
    many_body_features=[
        (0, 0, 1), (0, 0, 2),
        (0, 0, 0, 0, 1, 1)
    ],
    species=["W", "Re"]
)

pipeline.modifiers.append(TCEModifier(calc=calc))

for data in pipeline.frames:
    print(data.attributes)
```

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
- Tested on OVITO version [[VersionNumber]]

## Contact
- Jacob Jeffries: jwjeffr@clemson.edu
