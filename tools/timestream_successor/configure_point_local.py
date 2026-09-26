"""Reproduce the supplemental AppleClang/Homebrew cached build (not Unity/Spack)."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dependency-root',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[2];build=repo/'build';shim=build/'_pkgs/Glog';shim.mkdir(parents=True,exist_ok=True)
(shim/'GlogConfig.cmake').write_text('set(GLOG_FOUND TRUE)\nset(Glog_VERSION "fch_registered")\nmessage(STATUS "Use fch glog")\n')
values=dict(repo=str(repo),dependencies=str(a.dependency_root.resolve()),prefix=str(a.prefix.resolve()),python=sys.executable)
options=[s.format(**values) for s in json.loads((repo/'data/development/bright_point_base_v1/local-build-options.json').read_text())]
for opt in options:
 if opt.startswith('-DFETCHCONTENT_SOURCE_DIR_') and not Path(opt.split('=',1)[1]).is_dir():p.error('missing cached dependency: '+opt)
command=['cmake','-S',str(repo),'-B',str(build),*options];(build/'CONFIGURE.json').write_text(json.dumps(command,indent=2)+'\n');raise SystemExit(subprocess.call(command))
