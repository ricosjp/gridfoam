# Configuration templates

`simple.yaml`, `piso.yaml`, and `pimple.yaml` are Jinja templates for immersed
external flow. Render them before YAML parsing; the runner does not expand
`{{ ... }}` placeholders. Run from the repository root so mesh and output
paths resolve consistently.

| Parameter | Meaning |
| --- | --- |
| `mesh_path` | Immersed surface file path |
| `magU_ref` | Inlet speed and reference speed for force coefficients |
| `nu` | Kinematic viscosity |
| `A_ref`, `L_ref` | Reference area and length for SIMPLE force coefficients |

For example, render and validate a SIMPLE configuration:

```python
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined

from gridfoam.meta.config import GridfoamConfig

template = Environment(undefined=StrictUndefined).from_string(
    Path("assets/simple.yaml").read_text()
)
rendered = template.render(
    mesh_path="experiments/re_vs_cd/data/sphere.stl",
    magU_ref=1.0,
    nu=0.01,
    A_ref=1.0,
    L_ref=1.0,
)
config = GridfoamConfig.model_validate(yaml.safe_load(rendered))
```

Choose physical parameters, domain, resolution, output path, and device for
your case before running. The templates default to CUDA; use
`simulator.device: cpu` for CPU. Jinja2 is available in the development
environment through the documentation dependencies.

SIMPLE uses only `p` for pressure solves. PISO/PIMPLE optionally use `pFinal`
on the last inner corrector's last non-orthogonal pass, in each PIMPLE outer
iteration. Scalar initial and boundary values are numbers such as `0.0`;
vectors use three-element lists.

See the [configuration guide](../docs/source/user_guide/configuration.rst)
for boundary behavior and the [case guide](../docs/source/user_guide/running_cases.rst)
for runnable examples.
