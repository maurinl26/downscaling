# Docstring conventions

`karpos-downscaling` follows the **NumPy docstring style**
([numpydoc](https://numpydoc.readthedocs.io/)) for all public Python
docstrings. The Sphinx documentation uses `sphinx.ext.napoleon` to parse
this style and render API pages automatically.

This page codifies the conventions so that all contributors converge on the
same shape.

## What requires a docstring

| Object | Docstring required? | Notes |
|---|---|---|
| Module (`*.py`) | ✅ Yes | Top of file, describes purpose and key public APIs. |
| Public class | ✅ Yes | Describes the class. Constructor `__init__` docs may live in either the class or `__init__` docstring (we put them in the class). |
| Public method or function (no leading `_`) | ✅ Yes (with exceptions) | See exceptions below. |
| Private function (leading `_`) | ❌ Optional | Document if non-trivial. |
| Framework override (e.g. PyTorch Lightning `forward`, `training_step`, `configure_optimizers`) | ❌ Not required | The contract is defined by the framework. Document only when behavior is non-standard. |
| `main()` of CLI scripts | ❌ Not required | The ``argparse`` ``description=...`` and ``--help`` text serve as documentation. |
| Inner closures and ``lr_lambda``-style helpers | ❌ Optional | Document only if non-obvious. |

## NumPy style cheat sheet

A typical function docstring:

```python
def regrid_to_dem(
    da: xr.DataArray,
    dem: xr.DataArray,
    method: str = "linear",
) -> xr.DataArray:
    """Regrid a coarse field onto the fine DEM grid.

    Parameters
    ----------
    da : xarray.DataArray
        Coarse input field. Must have ``lat`` and ``lon`` coordinates.
    dem : xarray.DataArray
        Target DEM grid with ``lat`` and ``lon`` coordinates.
    method : {'linear', 'nearest', 'cubic'}, default 'linear'
        Interpolation method passed to ``scipy.interpolate``.

    Returns
    -------
    xarray.DataArray
        Field regridded onto the DEM grid, with the same units and
        attributes as ``da``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.

    Notes
    -----
    For physical fields like temperature, prefer a lapse-rate-aware
    regridding (see ``downscaling.shared.lapse_rate``).

    Examples
    --------
    >>> dem = xr.open_dataarray("srtm.nc")
    >>> t2m_coarse = xr.open_dataarray("cerra_t2m.nc")
    >>> t2m_fine = regrid_to_dem(t2m_coarse, dem)
    """
```

Key sections, in this order when applicable:

1. **Summary line** — one short sentence, imperative mood, fits on one line.
2. **Extended description** — optional, separated by a blank line.
3. **Parameters** — types follow numpydoc syntax (e.g. ``list of str``,
   ``xarray.DataArray``, ``{'a', 'b'}``).
4. **Returns** — same convention.
5. **Yields** — for generators.
6. **Raises** — exceptions explicitly raised by the function.
7. **Warnings / Warns** — runtime warnings.
8. **See Also** — cross-references to related functions.
9. **Notes** — implementation details, gotchas, scientific context.
10. **References** — bibliographic citations (we prefer BibTeX in
    `docs/references.bib` over inline references for reusability).
11. **Examples** — short executable doctest. Keep these minimal; full
    examples belong in the user guide.

## A few project-specific conventions

### Language

- **Code identifiers** in docstrings stay in English (parameter names,
  module names, class names).
- **Free text** in docstrings is in **English**, to maximize accessibility
  for international contributors and reviewers.
- French commentary in code comments (e.g. ``# observed sur pod ...``) is
  acceptable but should be **outside** docstrings.

### Scientific references

Cite scientific work via Sphinx + `sphinxcontrib-bibtex`. In a docstring,
use:

```
References
----------
.. [1] :cite:`perez2018film`
```

The BibTeX entry must exist in `docs/references.bib`.

### Math

For inline math, use LaTeX in a Notes section:

```
Notes
-----
The lapse-rate correction is

.. math::

    T_\\text{fine} = T_\\text{coarse} + \\gamma \\cdot (z_\\text{coarse} - z_\\text{fine})

where :math:`\\gamma = -6.5 \\times 10^{-3}` K/m is the default
environmental lapse rate.
```

### Units

State units explicitly in Parameters and Returns:

```
Parameters
----------
t2m : xarray.DataArray
    2-meter temperature in degrees Celsius.
elevation : xarray.DataArray
    Elevation in meters above sea level.
```

## Coverage and tooling

We track docstring coverage as a quality signal:

```bash
uv run python -c "
import ast, pathlib
root = pathlib.Path('downscaling')
total, with_doc = 0, 0
for p in root.rglob('*.py'):
    tree = ast.parse(p.read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name.startswith('_'): continue
            total += 1
            if ast.get_docstring(n): with_doc += 1
print(f'{with_doc}/{total} ({100*with_doc/total:.1f}%) documented')
"
```

Current state (June 2026):

- **Modules**: 100 % documented (49/49)
- **Public classes**: 100 % documented (42/42)
- **Public functions/methods**: ~52 %, with the gap concentrated on PyTorch
  Lightning hooks and CLI ``main()`` entry points (acceptable per our
  conventions above)

When opening a PR that adds a new public class or function, please ship the
docstring alongside the code. Reviewers will flag missing or incomplete
docstrings.

## See also

- [Contributing guidelines](contributing.md)
- [Code of conduct](code-of-conduct.md)
- [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Sphinx napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
