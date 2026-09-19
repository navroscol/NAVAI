import json, os, subprocess, sys, time
REPO, COMMIT = "https://github.com/navroscol/navros-ai", "@@COMMIT@@"
ROOT = "/tmp/navros-ai"
if not os.path.exists(ROOT):
    subprocess.run(["git", "clone", "-q", REPO, ROOT], check=True)
subprocess.run(["git", "-C", ROOT, "checkout", "-q", COMMIT], check=True)
sys.path.insert(0, ROOT)
OUT = "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
print("código:", REPO, "commit", COMMIT, flush=True)


def env_report():
    import torch
    info = dict(commit=COMMIT, torch=torch.__version__, cuda=torch.cuda.is_available(), n_gpu=torch.cuda.device_count(),
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], cpus=os.cpu_count())
    print(info, flush=True)
    json.dump(info, open(f"{OUT}/entorno.json", "w"), indent=1)
