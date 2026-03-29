import os
from pathlib import Path

# important to run GX on GPU without running out of memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from desc import set_device

set_device("gpu")

from desc.examples import get
from desc.external.gx import GX

REPO_ROOT = Path(__file__).resolve().parents[1]
GX_PATH = Path(os.environ.get("TI_OPT_GX_DIR", REPO_ROOT / "gx")).resolve()
GX_EXEC = os.environ.get("TI_OPT_GX_EXEC", "gx")
GX_INPUT_FILE = Path(
    os.environ.get("TI_OPT_GX_INPUT_FILE", "gx.in")
).resolve()
GX_LAUNCH = Path(os.environ.get("TI_OPT_GX_LAUNCH", REPO_ROOT / "gx-launch.sh")).resolve()

# DESC's GX objective always shells out via subprocess.run. The launch wrapper is
# therefore a prefix command for the child process, not an in-process library shim.
LAUNCH_CMD = [str(GX_LAUNCH)] if GX_LAUNCH.is_file() else None

eq = get("precise_qa")
obj = GX(
    eq,
    abs_step=1e-4,
    rel_step=0,
    processes=1,
    path=str(GX_PATH),
    npol=1,
    nzgrid=48,
    alpha=0.0,
    psi=0.5,
    sigma_Bxy=-1.0,
    # Template GX input file. DESC will patch geo_option to "eik" and replace
    # geo_file with the generated geometry path for each run.
    gx_input_file=str(GX_INPUT_FILE),
    launch_cmd=LAUNCH_CMD,
    gx_gpu=None,
    timeout=600,
    tmp_dir="tmp_GX",
    save_tmp=False,
    name="GX",
    **{"exec": GX_EXEC},
)
obj.build()
f = obj.compute(eq.params_dict)
print(f)
