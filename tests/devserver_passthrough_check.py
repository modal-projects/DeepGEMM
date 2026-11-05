"""Host-side check of the dev-loop knob passthrough in csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp (no GPU).

1. Every DG_MEGA_MOE_* name the explicit define lines read (std::getenv("DG_MEGA_MOE_...")) is in kMegaMoEHandledEnv,
   so the passthrough never emits a second, conflicting define for an explicit knob.
2. Extracts the DEVPASS block, compiles it with g++ and checks: handled names skipped, empty values skipped, unknown
   names forwarded verbatim and sorted, bad names / multi-line values rejected.
Usage: python tests/devserver_passthrough_check.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPP = os.path.join(ROOT, 'csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp')
src = open(HPP).read()
block = src[src.index('// DEVPASS_BEGIN'):src.index('// DEVPASS_END')]
handled = set(re.findall(r'"(DG_MEGA_MOE_[A-Z0-9_]+)"', block.split('static std::string get_mega_moe_passthrough_defines')[0]))
explicit = set(re.findall(r'getenv\("(DG_MEGA_MOE_[A-Z0-9_]+)"\)', src[src.index('// DEVPASS_END'):]))
missing = explicit - handled
assert not missing, f'explicit knobs missing from kMegaMoEHandledEnv: {sorted(missing)}'
# Python/host-side readers must be skipped too
py_names = set()
for d, _, fs in os.walk(os.path.join(ROOT, 'deep_gemm')):
    for f in fs:
        if f.endswith('.py'):
            py_names |= set(re.findall(r"'(DG_MEGA_MOE_[A-Z0-9_]+)'", open(os.path.join(d, f)).read()))
py_names |= set(re.findall(r"'(DG_MEGA_MOE_[A-Z0-9_]+)'", open(os.path.join(ROOT, 'tests/test_mega_moe.py')).read()))
assert not (py_names - handled), f'Python-read names missing from kMegaMoEHandledEnv: {sorted(py_names - handled)}'
print(f'ok: {len(explicit)} explicit + {len(py_names - explicit)} Python-only names handled')

cpp = r'''
#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#define DG_HOST_ASSERT(c) do { if (not (c)) throw std::runtime_error(#c); } while (0)
namespace deep_gemm {
''' + block + r'''
}
int main() {
    try { std::fputs(deep_gemm::get_mega_moe_passthrough_defines().c_str(), stdout); return 0; }
    catch (const std::exception& e) { std::fprintf(stdout, "THROW %s\n", e.what()); return 3; }
}
'''
with tempfile.TemporaryDirectory() as tmp:
    exe = os.path.join(tmp, 'pt')
    open(exe + '.cpp', 'w').write(cpp)
    subprocess.run(['g++', '-std=c++20', '-O0', '-o', exe, exe + '.cpp'], check=True)

    def run(env):
        base = {k: v for k, v in os.environ.items() if not k.startswith('DG_MEGA_MOE_')}
        p = subprocess.run([exe], env={**base, **env}, capture_output=True, text=True)
        return p.returncode, p.stdout

    rc, out = run({'DG_MEGA_MOE_IB': '1', 'DG_MEGA_MOE_IB_SKIP_CREDIT': '0', 'DG_MEGA_MOE_IB_TOLERANCE': '1e-3',
                   'DG_MEGA_MOE_ZZ_NEW': '3', 'DG_MEGA_MOE_AA_NEW': '0', 'DG_MEGA_MOE_EMPTY': '',
                   'DG_MEGA_MOE_DEV_HDR': '0x1234abcd', 'OTHER_DG_MEGA_MOE_X': '1'})
    assert rc == 0, out
    assert out == '#define DG_MEGA_MOE_AA_NEW 0\n#define DG_MEGA_MOE_DEV_HDR 0x1234abcd\n#define DG_MEGA_MOE_ZZ_NEW 3\n', repr(out)
    rc, out = run({})
    assert rc == 0 and out == '', repr(out)
    rc, out = run({'DG_MEGA_MOE_bad': '1'})
    assert rc == 3 and 'THROW' in out, out
    rc, out = run({'DG_MEGA_MOE_X': 'a\nb'})
    assert rc == 3 and 'THROW' in out, out
print('ok: passthrough behaviour')
