"""Download finished jobs' logs from the results Volume: fetch.py <cluster>/<job> ... (prints done/pending)."""
import modal, os, pathlib, sys
ENV = os.environ.get("MODAL_ENVIRONMENT") or None
st = modal.Dict.from_name("fused-ib-state", environment_name=ENV)
vol = modal.Volume.from_name("fused-ib-results", environment_name=ENV)
for cj in sys.argv[1:]:
    c, j = cj.split("/", 1)
    if not st.get(f"{c}/{j}/done"):
        print("pending", cj); continue
    out = pathlib.Path("results") / c / j
    out.mkdir(parents=True, exist_ok=True)
    for e in vol.listdir(f"/{c}/{j}"):
        name = pathlib.Path(e.path).name
        with open(out / name, "wb") as f:
            for chunk in vol.read_file(e.path):
                f.write(chunk)
    print("got", cj, {k: st.get(f"{c}/{j}/rank{k}") for k in range(8) if st.get(f"{c}/{j}/rank{k}") is not None})
