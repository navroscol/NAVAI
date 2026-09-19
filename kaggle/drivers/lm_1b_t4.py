# NAVROS-1B — entrenamiento en GPU T4 x2, por sesiones de ~11 h que se reanudan solas.
# Entradas: la salida de navros-datos-p0 (corpus) y, a partir de la 2.ª sesión, la salida de la sesión anterior
# (carpeta ckpt/ con latest.json). El commit, el preset y los hiperparámetros quedan en results/lm_1b.json.
#@@HEADER@@
PRESET = "@@PRESET@@"
TOTAL_TOKENS = @@TOKENS@@
if __name__ == "__main__":
    env_report()
    from navros.launch import launch
    from navros.lm import find_data_dirs
    dirs = find_data_dirs()
    print("datos:", dirs, "| checkpoints adjuntos:", sorted(__import__("glob").glob("/kaggle/input/**/latest.json", recursive=True)), flush=True)
    assert dirs, "adjunta la salida de navros-datos-p0"
    t0 = time.time()
    cfg = dict(preset=PRESET, data_dirs=dirs, T=1024, batch=256, tokens=TOTAL_TOKENS, muon_buf="bf16",
               lr_muon=0.02, lr_adam=1e-3, warmup=100, decay_frac=0.2, precision="fp16", device="cuda",
               eval_every=100, eval_seq=64, r_eval=[1, 2, 3, 4, 6, 8], log_every=5,
               ckpt_dir="/kaggle/working/ckpt", ckpt_every_min=45, tag=PRESET + "-v1")
    # 1.º sin recomputación (≈30 % más rápido si cabe); si se queda sin memoria, reanuda con recomputación.
    for extra in (dict(micro=1, grad_ckpt=False), dict(micro=2, grad_ckpt=True)):
        budget = 11 * 3600 - (time.time() - t0)
        code = launch(cfg | extra | dict(time_budget_s=budget), f"{OUT}/lm_1b.json", nproc=2, root=ROOT)
        print("salida", code, extra, flush=True)
        if code == 0:
            break
