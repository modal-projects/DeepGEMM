"""Write an NCCL topology XML for VM hosts whose GPUs and data NICs sit flat on per-domain root buses
(one B200 VM type: GPUs 0002:00:01-04 + mlx5_5-8 at 0002:00:09-0c, same for 0003). NCCL otherwise places
the NICs under the CPU with no PCIe path to any GPU ("Could not find any local path from gpu to net").
Each PCI domain gets one virtual bridge holding its GPUs and NICs. Usage: python nccl_topo.py OUT.xml"""
import os
import subprocess
import sys
from collections import defaultdict

out = sys.argv[1]
gpus = [line.split(",") for line in subprocess.run(
    ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
    capture_output=True, text=True, check=True).stdout.strip().splitlines()]
hcas = [h for h in os.environ.get("NCCL_IB_HCA", "").lstrip("=").split(",") if h]


def norm(busid):  # 00000002:00:01.0 -> 0002:00:01.0
    dom, rest = busid.strip().lower().split(":", 1)
    return f"{int(dom, 16):04x}:{rest}"


# Only VM hosts with GPUs on non-zero PCI domains need this; bare metal keeps NCCL's own topology
if all(norm(bus).startswith("0000:") for _, bus in gpus):
    print("nccl_topo: GPUs on PCI domain 0000, no topology override needed")
    sys.exit(3)

by_dom = defaultdict(lambda: {"gpus": [], "nics": []})
for idx, bus in gpus:
    b = norm(bus)
    by_dom[b.split(":")[0]]["gpus"].append(b)
for h in hcas:
    b = os.path.basename(os.path.realpath(f"/sys/class/infiniband/{h}/device")).lower()
    by_dom[b.split(":")[0]]["nics"].append((h, b))

lines = ['<system version="1">', '  <cpu numaid="0" arch="x86_64" vendor="GenuineIntel">']
for dom, d in sorted(by_dom.items()):
    if not d["gpus"] and not d["nics"]:
        continue
    lines.append(f'    <pci busid="{dom}:00:00.0" class="0x060400" link_speed="32.0 GT/s PCIe" link_width="16">')
    for b in d["gpus"]:
        lines.append(f'      <pci busid="{b}" class="0x030200" link_speed="32.0 GT/s PCIe" link_width="16"/>')
    for h, b in d["nics"]:
        lines.append(f'      <pci busid="{b}" class="0x020700" link_speed="32.0 GT/s PCIe" link_width="16">')
        lines.append(f'        <nic><net name="{h}" dev="-1"/></nic>')
        lines.append('      </pci>')
    lines.append('    </pci>')
lines += ['  </cpu>', '</system>']
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
