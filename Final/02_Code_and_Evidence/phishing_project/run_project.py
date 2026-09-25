"""Project launcher. Uses only Python's standard library until setup completes."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
VENV = ROOT / '.venv'
VENV_PYTHON = VENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
REQUIRED = {'numpy':'numpy', 'pandas':'pandas', 'scipy':'scipy',
            'sklearn':'scikit-learn', 'matplotlib':'matplotlib',
            'threadpoolctl':'threadpoolctl', 'IPython':'ipython',
            'ipykernel':'ipykernel', 'jupyterlab':'jupyterlab'}

def check_python():
    if not (3,11) <= sys.version_info[:2] <= (3,13):
        raise RuntimeError('This package supports Python 3.11, 3.12 or 3.13. '
                           'Install Python 3.12, then run Start_Mac.command again.')

def setup():
    check_python()
    if not VENV_PYTHON.exists():
        print('Creating an isolated project environment...', flush=True)
        subprocess.run([sys.executable, '-m', 'venv', str(VENV)], check=True)
    print('Installing project dependencies (internet needed on first setup)...', flush=True)
    subprocess.run([str(VENV_PYTHON), '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt')], check=True)
    # Local kernel specification: no global/user kernel registration necessary.
    subprocess.run([str(VENV_PYTHON), '-m', 'ipykernel', 'install', '--prefix', str(VENV),
                    '--name', 'prt565-phishing', '--display-name', 'PRT565 Phishing (.venv)'], check=True)
    subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve()), 'check'], check=True)
    print('Setup complete.', flush=True)

def check(require_notebook=True):
    check_python()
    print('Python:', sys.version.split()[0], 'Architecture:', platform.machine())
    print('Interpreter:', sys.executable)
    print('Project:', ROOT)
    problems=[]
    for module, distribution in REQUIRED.items():
        if not require_notebook and module in {"IPython", "ipykernel", "jupyterlab"}:
            continue
        try:
            importlib.import_module(module)
            print(distribution+':', importlib.metadata.version(distribution))
        except Exception as exc:
            problems.append(module+': '+str(exc))
    if problems:
        raise RuntimeError('Dependencies cannot load:\n'+'\n'.join(problems)+
                           '\nRun Start_Mac.command to install them into .venv. '
                           'In VS Code, select the .venv interpreter/kernel.')
    sys.path.insert(0,str(ROOT))
    from train import load_data
    X,y,groups,source=load_data()
    print('Dataset loads:',X.shape,'phishing records:',int(y.sum()))
    print('Setup check passed.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','check','notebook','train','deep'],nargs='?',default='check')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='setup':
        setup(); return
    check(require_notebook=args.action not in ('train','deep'))
    if args.action=='train':
        from train import main as train
        train()
    elif args.action=='deep':
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError('PyTorch is not installed. Run: .venv/bin/python -m pip install -r requirements-deep-learning.txt')
        from deep_learning_extension import main as deep
        deep()
    elif args.action=='notebook':
        subprocess.run([sys.executable,'-m','jupyterlab','--ServerApp.root_dir='+str(ROOT),
                        str(ROOT/'Phishing_Lab.ipynb')],check=True)

if __name__=='__main__':
    try: main()
    except (Exception,KeyboardInterrupt) as exc:
        report=ROOT/'startup_error.txt'
        report.write_text('Python: '+sys.version+'\nPlatform: '+platform.platform()+
                          '\nAction: '+str(sys.argv[1:])+'\n\n'+traceback.format_exc(),encoding='utf-8')
        print('\nCould not complete startup:',exc,file=sys.stderr)
        print('Error details saved in '+str(report),file=sys.stderr)
        sys.exit(1)
