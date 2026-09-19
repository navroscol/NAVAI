# NAVROS — benchmark de NAVROS-1B en GPU T4 x2 (tokens sintéticos): memoria y velocidad reales
# del entrenador distribuido (DDP + Muon repartido por dueño + fp16), recurrente y fijo.
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.launch import launch
    base = dict(synthetic=True, T=1024, batch=64, warmup=2, precision="fp16", device="cuda", grad_ckpt=True,
                eval_every=10**9, eval_seq=8, r_eval=[1, 2, 4], log_every=1, lr_muon=0.02, lr_adam=0.003)
    for name, kw in [("rec_m2", dict(preset="navros-1b", micro=2)),
                     ("fix_m2", dict(preset="navros-1b-fix", micro=2)),
                     ("rec_m4", dict(preset="navros-1b", micro=4))]:
        cfg = base | kw | dict(tokens=6 * 64 * 1024, tag=name)
        print(f"==== {name}: {cfg}", flush=True)
        code = launch(cfg, f"{OUT}/bench_{name}.json", nproc=2, root=ROOT)
        print(f"==== {name}: salida {code}", flush=True)
