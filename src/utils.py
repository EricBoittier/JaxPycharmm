<<<<<<< HEAD
=======
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import orbax

DTYPE = jnp.float32
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

def get_files(path: str) -> List[Path]:
    """Get sorted directory paths excluding tmp directories."""
    dirs = list(Path(path).glob("*/"))
    dirs.sort(key=lambda x: int(str(x).split("/")[-1].split("-")[-1]))
    return [_ for _ in dirs if "tmp" not in str(_)]

def get_last(path: str) -> Path:
    """Get the last checkpoint directory."""
    dirs = get_files(path)
    if "tmp" in str(dirs[-1]):
        dirs.pop()
    return dirs[-1]

def get_params_model(restart: str, natoms: int = None):
    """Load parameters and model from checkpoint."""
    restored = orbax_checkpointer.restore(restart)
    print(f"Restoring from {restart}")
    
    modification_time = os.path.getmtime(restart)
    modification_date = datetime.fromtimestamp(modification_time)
    print(f"The file was last modified on: {modification_date}")
    print("Restored keys:", restored.keys())
    
    params = restored["params"]
    
    if "model_attributes" not in restored.keys():
        return params, None
        
    kwargs = _process_model_attributes(restored["model_attributes"], natoms)
    model = EF(**kwargs)
    print(model)
    return params, model

def _process_model_attributes(attrs: Dict[str, Any], natoms: int = None) -> Dict[str, Any]:
    """Process model attributes from checkpoint."""
    kwargs = attrs.copy()
    
    int_fields = ["features", "max_degree", "num_iterations", "num_basis_functions",
                 "natoms", "n_res", "max_atomic_number"]
    float_fields = ["cutoff", "total_charge"]
    bool_fields = ["charges"]
    
    for field in int_fields:
        kwargs[field] = int(kwargs[field])
    for field in float_fields:
        kwargs[field] = float(kwargs[field])
    for field in bool_fields:
        kwargs[field] = bool(kwargs[field])
        
    kwargs["debug"] = []
    if natoms is not None:
        kwargs["natoms"] = natoms
        
    return kwargs
>>>>>>> Snippet
<<<<<<< HEAD
=======
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import orbax

DTYPE = jnp.float32
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

def get_files(path: str) -> List[Path]:
    """Get sorted directory paths excluding tmp directories."""
    dirs = list(Path(path).glob("*/"))
    dirs.sort(key=lambda x: int(str(x).split("/")[-1].split("-")[-1]))
    return [_ for _ in dirs if "tmp" not in str(_)]

def get_last(path: str) -> Path:
    """Get the last checkpoint directory."""
    dirs = get_files(path)
    if "tmp" in str(dirs[-1]):
        dirs.pop()
    return dirs[-1]

def get_params_model(restart: str, natoms: int = None):
    """Load parameters and model from checkpoint."""
    restored = orbax_checkpointer.restore(restart)
    print(f"Restoring from {restart}")
    
    modification_time = os.path.getmtime(restart)
    modification_date = datetime.fromtimestamp(modification_time)
    print(f"The file was last modified on: {modification_date}")
    print("Restored keys:", restored.keys())
    
    params = restored["params"]
    
    if "model_attributes" not in restored.keys():
        return params, None
        
    kwargs = _process_model_attributes(restored["model_attributes"], natoms)
    model = EF(**kwargs)
    print(model)
    return params, model

def _process_model_attributes(attrs: Dict[str, Any], natoms: int = None) -> Dict[str, Any]:
    """Process model attributes from checkpoint."""
    kwargs = attrs.copy()
    
    int_fields = ["features", "max_degree", "num_iterations", "num_basis_functions",
                 "natoms", "n_res", "max_atomic_number"]
    float_fields = ["cutoff", "total_charge"]
    bool_fields = ["charges"]
    
    for field in int_fields:
        kwargs[field] = int(kwargs[field])
    for field in float_fields:
        kwargs[field] = float(kwargs[field])
    for field in bool_fields:
        kwargs[field] = bool(kwargs[field])
        
    kwargs["debug"] = []
    if natoms is not None:
        kwargs["natoms"] = natoms
        
    return kwargs
>>>>>>> Snippet
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import jax.numpy as jnp
import orbax

DTYPE = jnp.float32
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def get_files(path: str) -> List[Path]:
    """Get sorted directory paths excluding tmp directories."""
    dirs = list(Path(path).glob("*/"))
    dirs.sort(key=lambda x: int(str(x).split("/")[-1].split("-")[-1]))
    return [_ for _ in dirs if "tmp" not in str(_)]


def get_last(path: str) -> Path:
    """Get the last checkpoint directory."""
    dirs = get_files(path)
    if "tmp" in str(dirs[-1]):
        dirs.pop()
    return dirs[-1]


def get_params_model(restart: str, natoms: int = None):
    """Load parameters and model from checkpoint."""
    restored = orbax_checkpointer.restore(restart)
    print(f"Restoring from {restart}")

    modification_time = os.path.getmtime(restart)
    modification_date = datetime.fromtimestamp(modification_time)
    print(f"The file was last modified on: {modification_date}")
    print("Restored keys:", restored.keys())

    params = restored["params"]

    if "model_attributes" not in restored.keys():
        return params, None

    kwargs = _process_model_attributes(restored["model_attributes"], natoms)
    model = EF(**kwargs)
    print(model)
    return params, model


def _process_model_attributes(
    attrs: Dict[str, Any], natoms: int = None
) -> Dict[str, Any]:
    """Process model attributes from checkpoint."""
    kwargs = attrs.copy()

    int_fields = [
        "features",
        "max_degree",
        "num_iterations",
        "num_basis_functions",
        "natoms",
        "n_res",
        "max_atomic_number",
    ]
    float_fields = ["cutoff", "total_charge"]
    bool_fields = ["charges"]

    for field in int_fields:
        kwargs[field] = int(kwargs[field])
    for field in float_fields:
        kwargs[field] = float(kwargs[field])
    for field in bool_fields:
        kwargs[field] = bool(kwargs[field])

    kwargs["debug"] = []
    if natoms is not None:
        kwargs["natoms"] = natoms

    return kwargs
