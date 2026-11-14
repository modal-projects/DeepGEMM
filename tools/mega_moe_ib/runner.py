"""Long-lived clustered job runner: hold N B300 nodes with RDMA and run submitted jobs on all of them.

Start (holds the cluster until --hours runs out or `submit.py --stop`):
    modal run --detach runner.py --cluster b300_2n_a

Clusters (see README.md): b300_2n_a/b300_4n_a and b300_2n_b/b300_4n_b (B300, ConnectX-8 InfiniBand), b300_2n_u/b300_4n_u (unpinned),
gb200_16_a/gb200_32_a (GB200 trays). Each holds its nodes for up to 24 h (Modal's function timeout), then exits;
start it again for another day.

Deployment settings come from the environment (README.md): FUSED_IB_APP (app name), FUSED_IB_CLOUDS
(JSON {cluster: cloud} pins), and Modal's own MODAL_PROFILE / MODAL_ENVIRONMENT.

Protocol (all state in Modal objects named fused-ib-*):
  Queue  fused-ib-jobs, partition <cluster>     submit.py puts job specs; rank 0 takes them.
  Dict   fused-ib-state                         rank 0 publishes "<cluster>/current" = spec;
                                                every rank writes "<cluster>/<job>/rank<r>" = exit code;
                                                "<cluster>/heartbeat" = {time, ips, provider}.
  Volume fused-ib-results                       /<cluster>/<job>/rank<r>.log, plus files the job
                                                writes under $RESULTS_DIR.
A job spec: {"id", "script" (bash), "files" {name: text}, "timeout_s"}. Each rank runs the script in
/work/<job> with NODE_RANK, NNODES, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE, RESULTS_DIR set, plus the
fabric env (EFA plugin paths on AWS). A job runs on every rank; rank 0 waits for all ranks before
taking the next job, so jobs never overlap.
"""

import os as _os
_os.environ.setdefault("MODAL_FUNCTION_RUNTIME", "runc")  # every cluster runs on runc (GDAKI, GB200); set before modal is imported
import glob
import json
import os
import queue
import subprocess
import threading
import time

import modal
import modal.experimental

GPUS = 8
APP = os.environ.get("FUSED_IB_APP", "fused-ib-runner")
QUEUE, STATE, RESULTS = "fused-ib-jobs", "fused-ib-state", "fused-ib-results"
MAX_HOURS = 23.5
POLL_S = 2.0
# Cloud pin per cluster, e.g. FUSED_IB_CLOUDS='{"b300_4n_a": "<cloud>", "b300_4n_b": "<cloud>"}'. A cluster without an entry is
# not pinned (Modal places it wherever the GPU type with RDMA is available).
CLOUDS = json.loads(os.environ.get("FUSED_IB_CLOUDS", "{}"))


def _cloud(cluster: str) -> dict:
    return {"cloud": CLOUDS[cluster]} if CLOUDS.get(cluster) else {}

def _gpus_per_node() -> int:
    try:
        return len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
    except Exception:
        return GPUS


image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "git", "build-essential", "cmake", "ninja-build", "ibverbs-utils", "rdma-core",
        "infiniband-diags", "perftest", "pciutils", "iproute2", "numactl", "libibverbs-dev",
        "libdw-dev", "libelf-dev",
    )
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("nvidia-nccl-cu13==2.32.3", "numpy", "ninja", "packaging", "setuptools", "wheel")
)
app = modal.App(APP, image=image)
results_vol = modal.Volume.from_name(RESULTS, create_if_missing=True)
jobs_q = modal.Queue.from_name(QUEUE, create_if_missing=True)
state = modal.Dict.from_name(STATE, create_if_missing=True)


def _sh(cmd: str) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (p.stdout + p.stderr).strip()


def _fabric_env() -> dict:
    env = dict(os.environ)
    if os.environ.get("FI_PROVIDER") == "efa":
        # aws-ofi-nccl dlopens libcudart.so; pip wheels ship only libcudart.so.<major>.
        os.makedirs("/tmp/cudart", exist_ok=True)
        rt = sorted(glob.glob("/usr/local/lib/python3.12/site-packages/nvidia/**/libcudart.so*", recursive=True))
        if rt and not os.path.exists("/tmp/cudart/libcudart.so"):
            os.symlink(rt[0], "/tmp/cudart/libcudart.so")
        libs = "/tmp/cudart:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib"
        env["LD_LIBRARY_PATH"] = libs + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _run_job(cluster: str, spec: dict, rank: int, ips: list[str]) -> int:
    job = spec["id"]
    work = f"/work/{job}"
    out_dir = f"/results/{cluster}/{job}"
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for name, text in spec.get("files", {}).items():
        path = os.path.join(work, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    with open(f"{work}/job.sh", "w") as f:
        f.write(spec["script"])
    env = _fabric_env()
    env.update(
        NODE_RANK=str(rank), NNODES=str(len(ips)), MASTER_ADDR=ips[0], MASTER_PORT="29500",
        GPUS_PER_NODE=str(_gpus_per_node()), RESULTS_DIR=out_dir, JOB_ID=job, CLUSTER=cluster,
    )
    log = f"{out_dir}/rank{rank}.log"
    t0 = time.time()
    with open(log, "w") as f:
        try:
            code = subprocess.run(
                ["bash", "job.sh"], cwd=work, env=env, stdout=f, stderr=subprocess.STDOUT,
                timeout=spec.get("timeout_s", 3600),
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[runner] job timed out after {spec.get('timeout_s', 3600)} s\n")
            code = 124
        f.write(f"\n[runner] rank {rank} exit {code} after {time.time() - t0:.1f} s\n")
    results_vol.commit()
    state[f"{cluster}/{job}/rank{rank}"] = code
    return code


def _serve(cluster: str, hours: float):
    info = modal.experimental.get_cluster_info()
    rank, ips = info.rank, list(info.container_ipv4_ips)
    provider = os.environ.get("MODAL_CLOUD_PROVIDER", "")
    nics = _sh("echo kernel=$(uname -r) runtime=$( [ -d /proc/gvisor ] || dmesg 2>/dev/null | grep -qi gvisor && echo gvisor || echo runc-or-unknown); "
               "for d in /sys/class/infiniband/*; do echo $(basename $d) "
               "$(basename $(readlink -f $d/device/driver)) $(cat $d/ports/1/rate 2>/dev/null); done")
    print(f"[{cluster} rank {rank}] up: provider={provider} region={os.environ.get('MODAL_REGION')} "
          f"ips={ips}\n{nics}", flush=True)
    if cluster in FABRIC_GUARD and not any("mlx5" in l and "800 Gb" in l for l in nics.splitlines()):
        # unpinned request landed on a non-CX-8 fabric (e.g. AWS EFA): release the nodes at once; the watchdog retries
        if rank == 0:
            n = (state.get(f"{cluster}/rejects") or 0) + 1
            state[f"{cluster}/rejects"] = n
            state[f"{cluster}/last_reject"] = {"time": time.time(), "provider": provider, "nics": nics}
        print(f"[{cluster} rank {rank}] fabric guard: no 800G mlx5 NIC (provider={provider}); releasing", flush=True)
        return
    deadline = time.time() + min(hours, MAX_HOURS) * 3600
    last_seq = state.get(f"{cluster}/seq", 0) if rank else None
    if rank == 0:
        state[f"{cluster}/seq"] = last_seq = state.get(f"{cluster}/seq", 0)
        busy = {"job": None}
        stop_beat = threading.Event()

        def beat():  # heartbeats continue while a job runs; stop when _serve returns
            while time.time() < deadline and not stop_beat.is_set():
                try:
                    state[f"{cluster}/heartbeat"] = {"time": time.time(), "ips": ips, "provider": provider,
                                                     "nodes": len(ips), "nics": nics, "deadline": deadline,
                                                     "busy": busy["job"]}
                except Exception as e:
                    print(f"[{cluster}] heartbeat failed: {e}", flush=True)
                time.sleep(20)

        threading.Thread(target=beat, daemon=True).start()
    try:
        _serve_loop(cluster, rank, ips, deadline, last_seq, busy if rank == 0 else None)
    finally:
        if rank == 0:
            stop_beat.set()
            state[f"{cluster}/heartbeat"] = {"time": 0, "ips": ips, "provider": provider, "nodes": len(ips),
                                             "nics": nics, "deadline": deadline, "busy": None}


def _serve_loop(cluster, rank, ips, deadline, last_seq, busy):
    while time.time() < deadline:
        if rank == 0:
            busy["job"] = None
            try:
                spec = jobs_q.get(partition=cluster, timeout=30)
            except queue.Empty:  # Modal's Queue.get raises on timeout instead of returning None
                continue
            if spec is None:
                continue
            if spec.get("stop"):
                state[f"{cluster}/current"] = {"seq": last_seq + 1, "stop": True}
                state[f"{cluster}/seq"] = last_seq + 1  # so the next runner's ranks skip this stop
                print(f"[{cluster}] stop requested", flush=True)
                return
            last_seq += 1
            spec["seq"] = last_seq
            state[f"{cluster}/current"] = spec
            state[f"{cluster}/seq"] = last_seq
            busy["job"] = spec["id"]
            print(f"[{cluster}] job {spec['id']} (seq {last_seq})", flush=True)
            _run_job(cluster, spec, rank, ips)
            # wait for every rank before taking the next job
            t_wait = time.time() + spec.get("timeout_s", 3600) + 120
            while time.time() < t_wait and not all(
                f"{cluster}/{spec['id']}/rank{r}" in state for r in range(len(ips))
            ):
                time.sleep(POLL_S)
            state[f"{cluster}/{spec['id']}/done"] = True
        else:
            cur = state.get(f"{cluster}/current")
            if not cur or cur.get("seq", 0) <= last_seq:
                time.sleep(POLL_S)
                continue
            last_seq = cur["seq"]
            if cur.get("stop"):
                return
            _run_job(cluster, cur, rank, ips)
    print(f"[{cluster} rank {rank}] deadline reached", flush=True)


# exclude_reserved_capacity: never land on reserved capacity. Every cluster function (B300, B200, GB200) inherits it
# from FN.
FN = dict(gpu=f"B300:{GPUS}", timeout=24 * 3600, cpu=64, memory=512 * 1024,
          volumes={"/results": results_vol}, experimental_options={"exclude_reserved_capacity": True})


# Proxy Gin needs CPU for NCCL proxy threads (8 GPU processes x several busy-polling threads); 32 CPUs gave
# ms-scale, highly variable signal latency on b300_2n_a. B300 bare-metal hosts have ~115 CPUs; Modal caps at 64.
FN_BIGCPU = dict(FN, cpu=64)  # Modal max per function


@app.function(**FN_BIGCPU, **_cloud("b300_2n_a"))
@modal.clustered(size=2, rdma=True)
def b300_2n_a(hours: float = MAX_HOURS):
    _serve("b300_2n_a", hours)


@app.function(**FN, **_cloud("b300_4n_a"))
@modal.clustered(size=4, rdma=True)
def b300_4n_a(hours: float = MAX_HOURS):
    _serve("b300_4n_a", hours)


@app.function(**FN, **_cloud("b300_2n_b"))
@modal.clustered(size=2, rdma=True)
def b300_2n_b(hours: float = MAX_HOURS):
    _serve("b300_2n_b", hours)


# Extra 2-node B300 clusters (more capacity for parallel A/Bs)
@app.function(**FN_BIGCPU, **_cloud("b300_2n_c"))
@modal.clustered(size=2, rdma=True)
def b300_2n_c(hours: float = MAX_HOURS):
    _serve("b300_2n_c", hours)


@app.function(**FN, **_cloud("b300_2n_d"))
@modal.clustered(size=2, rdma=True)
def b300_2n_d(hours: float = MAX_HOURS):
    _serve("b300_2n_d", hours)


@app.function(**FN, **_cloud("b300_4n_b"))
@modal.clustered(size=4, rdma=True)
def b300_4n_b(hours: float = MAX_HOURS):
    _serve("b300_4n_b", hours)


@app.function(**FN)
@modal.clustered(size=2, rdma=True)
def b300_2n_u(hours: float = MAX_HOURS):
    _serve("b300_2n_u", hours)


# Unpinned 4-node B300 that keeps only an 800G ConnectX fabric: the fabric guard in _serve returns at once on any
# other fabric, and the watchdog respawns it.
FABRIC_GUARD = {"b300_4n_g"}


@app.function(**FN)
@modal.clustered(size=4, rdma=True)
def b300_4n_g(hours: float = MAX_HOURS):
    _serve("b300_4n_g", hours)


@app.function(**FN)
@modal.clustered(size=4, rdma=True)
def b300_4n_u(hours: float = MAX_HOURS):
    _serve("b300_4n_u", hours)


# B200 alternatives with mlx5-family RDMA (B300 and B200 are both fine for this work).
B200 = dict(FN, gpu=f"B200:{GPUS}")


# b200_2n_a disabled 2026-11-06: deploy fails (no worker type supports gpu_b200 on its cloud); plain function, not deployed
def b200_2n_a(hours: float = MAX_HOURS):
    _serve("b200_2n_a", hours)


@app.function(**B200, **_cloud("b200_2n_b"))
@modal.clustered(size=2, rdma=True)
def b200_2n_b(hours: float = MAX_HOURS):
    _serve("b200_2n_b", hours)


@app.function(**B200, **_cloud("b200_2n_c"))
@modal.clustered(size=2, rdma=True)
def b200_2n_c(hours: float = MAX_HOURS):
    _serve("b200_2n_c", hours)


@app.function(**B200, **_cloud("b200_2n_d"))
@modal.clustered(size=2, rdma=True)
def b200_2n_d(hours: float = MAX_HOURS):
    _serve("b200_2n_d", hours)


GB200 = dict(FN, gpu="GB200:4")  # bare-metal GB200 trays, 4 GPUs each (Grace arm64 host, runc only)
# aarch64 image for the Grace hosts:
# the base layer and every build step run on a GB200 builder (gpu_config / gpu="GB200"), and the image ends
# with a GPU-built layer that puts `python` on PATH so Modal can detect it
from modal_proto import api_pb2 as _api_pb2
BUILD_GPU = "GB200"
image_gb = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04",
                              gpu_config=_api_pb2.GPUConfig(gpu_type=BUILD_GPU, count=1))
    .dockerfile_commands("ENTRYPOINT []", gpu=BUILD_GPU)
    .run_commands(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv python3-dev "
        "git build-essential cmake ninja-build ibverbs-utils rdma-core infiniband-diags perftest pciutils iproute2 "
        "numactl libibverbs-dev libdw-dev libelf-dev",
        'P=$(command -v python3); ln -sf $P /usr/local/bin/python3; ln -sf $P /usr/local/bin/python; '
        "ln -sf $(command -v pip3) /usr/local/bin/pip",
        gpu=BUILD_GPU)
    .env({"PIP_BREAK_SYSTEM_PACKAGES": "1"})
    .run_commands("python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130", gpu=BUILD_GPU)
    .run_commands("python -m pip install nvidia-nccl-cu13==2.32.3 numpy ninja packaging setuptools wheel", gpu=BUILD_GPU)
    .run_commands("python --version", gpu=BUILD_GPU)
)


@app.function(**GB200, **_cloud("gb200_16_a"), image=image_gb)
@modal.experimental.clustered(size=4, rdma=True, fabric_size=4)  # one NVL72 rack
def gb200_16_a(hours: float = MAX_HOURS):
    _serve("gb200_16_a", hours)


@app.function(**GB200, **_cloud("gb200_32_a"), image=image_gb)
@modal.experimental.clustered(size=8, rdma=True, fabric_size=8)  # all 8 trays in one NVL72 rack
def gb200_32_a(hours: float = MAX_HOURS):
    _serve("gb200_32_a", hours)


# Two more 16-GPU GB200 clusters (work in parallel)
@app.function(**GB200, **_cloud("gb200_16_b"), image=image_gb)
@modal.experimental.clustered(size=4, rdma=True, fabric_size=4)
def gb200_16_b(hours: float = MAX_HOURS):
    _serve("gb200_16_b", hours)


@app.function(**GB200, **_cloud("gb200_16_c"), image=image_gb)
@modal.experimental.clustered(size=4, rdma=True, fabric_size=4)
def gb200_16_c(hours: float = MAX_HOURS):
    _serve("gb200_16_c", hours)


# A second 32-GPU single-rack GB200 cluster
@app.function(**GB200, **_cloud("gb200_32_b"), image=image_gb)
@modal.experimental.clustered(size=8, rdma=True, fabric_size=8)
def gb200_32_b(hours: float = MAX_HOURS):
    _serve("gb200_32_b", hours)


CLUSTERS = {"gb200_16_a": gb200_16_a, "gb200_32_a": gb200_32_a, "gb200_16_b": gb200_16_b, "gb200_16_c": gb200_16_c, "gb200_32_b": gb200_32_b, "b300_2n_a": b300_2n_a, "b300_2n_c": b300_2n_c, "b300_4n_a": b300_4n_a, "b300_2n_b": b300_2n_b, "b300_2n_d": b300_2n_d, "b300_4n_b": b300_4n_b, "b300_2n_u": b300_2n_u, "b300_4n_u": b300_4n_u, "b300_4n_g": b300_4n_g,
            "b200_2n_b": b200_2n_b, "b200_2n_c": b200_2n_c, "b200_2n_d": b200_2n_d}
# One runner per group is kept alive. In a group with several clusters, all are requested until one
# comes up; then the others' pending requests are cancelled.
DEFAULT_GROUPS = [["b300_2n_u"], ["b300_2n_a"], ["b300_2n_b"], ["b200_2n_b"], ["b200_2n_c", "b200_2n_d"], ["gb200_16_a"], ["gb200_32_a"], ["gb200_16_b"], ["gb200_16_c"], ["gb200_32_b"]]  # EFA, 800G CX-8 B300 x2, 400G IB B200 x2, 400G RoCE B200
STALE_S = 180


def _call_state(call_id: str | None) -> str:
    """'none' | 'running' (pending or running) | 'finished'."""
    if not call_id:
        return "none"
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "finished"
    except TimeoutError:
        return "running"
    except Exception:
        return "finished"  # failed, cancelled or expired


def _alive(cluster: str) -> bool:
    hb = state.get(f"{cluster}/heartbeat")
    if not hb:
        return False
    if time.time() - hb["time"] < STALE_S:
        return True
    # Runners started before the heartbeat thread only beat between jobs: a job still inside its
    # timeout counts as alive.
    cur = state.get(f"{cluster}/current")
    if cur and not cur.get("stop") and not state.get(f"{cluster}/{cur['id']}/done"):
        return time.time() - hb["time"] < cur.get("timeout_s", 3600) + STALE_S
    return False


def _requeue_orphan(cluster: str):
    cur = state.get(f"{cluster}/current")
    if not cur or cur.get("stop") or state.get(f"{cluster}/{cur['id']}/done"):
        return
    if state.get(f"{cluster}/{cur['id']}/requeued"):
        state[f"{cluster}/{cur['id']}/done"] = True  # second loss: give up so the client stops waiting
        state[f"{cluster}/{cur['id']}/lost"] = True
        return
    spec = {k: v for k, v in cur.items() if k != "seq"}
    jobs_q.put(spec, partition=cluster)
    state[f"{cluster}/{cur['id']}/requeued"] = True
    print(f"[watchdog] requeued orphaned job {cur['id']} on {cluster}", flush=True)


@app.function(schedule=modal.Cron("*/5 * * * *"), timeout=300)
def watchdog():
    if state.get("watchdog/paused"):
        return
    groups = state.get("watchdog/groups", DEFAULT_GROUPS)
    log = []
    for group in groups:
        group = [c for c in group if not state.get(f"{c}/disabled")]
        alive = [c for c in group if _alive(c)]
        calls = {c: state.get(f"{c}/call") for c in group}
        if alive:
            for c in group:
                if c not in alive and _call_state(calls[c]) == "running":
                    modal.FunctionCall.from_id(calls[c]).cancel()
                    state[f"{c}/call"] = None
                    log.append(f"cancelled {c} ({alive[0]} is up)")
            continue
        for c in group:
            if _call_state(calls[c]) == "running":
                # still waiting for capacity, or up but heartbeat stale (hung): give it 2 stale windows
                hb = state.get(f"{c}/heartbeat")
                spawned = state.get(f"{c}/spawned", 0)
                if hb and hb["time"] > spawned and time.time() - hb["time"] > 2 * STALE_S:
                    modal.FunctionCall.from_id(calls[c]).cancel()
                    log.append(f"cancelled hung {c}")
                else:
                    continue
            _requeue_orphan(c)
            call = CLUSTERS[c].spawn(MAX_HOURS)
            state[f"{c}/call"] = call.object_id
            state[f"{c}/spawned"] = time.time()
            log.append(f"spawned {c} ({call.object_id})")
    state["watchdog/last"] = {"time": time.time(), "log": log}
    print(f"[watchdog] {log or 'all groups healthy or pending'}", flush=True)


@app.local_entrypoint()
def main(cluster: str, hours: float = MAX_HOURS):
    CLUSTERS[cluster].remote(hours)
