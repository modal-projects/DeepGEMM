"""Submit a job to a held cluster (see runner.py) and wait for its logs.

    python submit.py --cluster b300_2n_a --script jobs/devserver.sh --file jobs/devserver.py ...
    python submit.py --cluster b300_2n_a --status
    python submit.py --cluster b300_2n_a --stop

The Modal profile and environment are Modal's own MODAL_PROFILE / MODAL_ENVIRONMENT (unset = your defaults).

Logs land in ./results/<cluster>/<job>/rank<r>.log (and anything the job wrote to $RESULTS_DIR).
"""

import argparse
import os
import pathlib
import sys
import time
import uuid

import modal

QUEUE, STATE, RESULTS = "fused-ib-jobs", "fused-ib-state", "fused-ib-results"
ENV = os.environ.get("MODAL_ENVIRONMENT") or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--script", help="bash script run on every node in the job's work dir")
    ap.add_argument("--file", action="append", default=[], help="file to ship next to the script")
    ap.add_argument("--timeout", type=int, default=3600, help="per-job timeout, seconds")
    ap.add_argument("--name", default="", help="job id prefix")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL exported before the script runs")
    a = ap.parse_args()

    q = modal.Queue.from_name(QUEUE, environment_name=ENV, create_if_missing=True)
    st = modal.Dict.from_name(STATE, environment_name=ENV, create_if_missing=True)
    vol = modal.Volume.from_name(RESULTS, environment_name=ENV, create_if_missing=True)

    hb = st.get(f"{a.cluster}/heartbeat")
    if a.status:
        if not hb:
            print(f"{a.cluster}: no runner has reported")
            return
        age = time.time() - hb["time"]
        print(f"{a.cluster}: heartbeat {age:.0f} s ago, provider {hb['provider']}, {hb['nodes']} nodes, "
              f"{(hb['deadline'] - time.time()) / 3600:.1f} h left, queued {q.len(partition=a.cluster)}\n{hb['nics']}")
        return
    if a.stop:
        st[f"{a.cluster}/disabled"] = True  # keep the watchdog from relaunching it
        q.put({"stop": True}, partition=a.cluster)
        call = st.get(f"{a.cluster}/call")
        if call and not hb:
            modal.FunctionCall.from_id(call).cancel()  # still waiting for capacity
        print(f"stop queued for {a.cluster}; watchdog disabled for it (--start to re-enable)")
        return
    if a.start:
        st[f"{a.cluster}/disabled"] = False
        print(f"{a.cluster} re-enabled; the watchdog launches it within 5 minutes")
        return
    if not a.script:
        sys.exit("--script is required")
    if not hb or time.time() - hb["time"] > 300:
        print(f"warning: {a.cluster} has no recent heartbeat; the job waits in the queue until a runner starts",
              file=sys.stderr)

    job = f"{a.name or pathlib.Path(a.script).stem}-{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    exports = "".join(f"export {kv}\n" for kv in a.env)
    spec = {"id": job, "script": exports + pathlib.Path(a.script).read_text(), "timeout_s": a.timeout,
            "files": {pathlib.Path(f).name: pathlib.Path(f).read_text() for f in a.file}}
    q.put(spec, partition=a.cluster)
    print(f"queued {job} on {a.cluster}", flush=True)
    if a.no_wait:
        return
    deadline = time.time() + a.timeout + 3600
    while time.time() < deadline and not st.get(f"{a.cluster}/{job}/done"):
        time.sleep(5)
    if not st.get(f"{a.cluster}/{job}/done"):
        sys.exit(f"{job}: not done before the wait deadline")
    if st.get(f"{a.cluster}/{job}/lost"):
        sys.exit(f"{job}: lost twice to runner crashes; see the runner logs")
    out = pathlib.Path("results") / a.cluster / job
    out.mkdir(parents=True, exist_ok=True)
    codes = {}
    for entry in vol.listdir(f"/{a.cluster}/{job}"):
        name = pathlib.Path(entry.path).name
        with open(out / name, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
    nodes = (hb or {}).get("nodes", 0)
    for r in range(nodes):
        codes[r] = st.get(f"{a.cluster}/{job}/rank{r}")
    print(f"{job}: exit codes {codes}; results in {out}")
    sys.exit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()
