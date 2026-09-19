# NAVROS — razonador v2: el mismo protocolo que v1 pero SIN RoPE (solo ábaco / posición con desplazamiento),
# más el barrido controlado de ancho del bucle (C = 64, 128, 256). GPU T4 x2.
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.experiments import reasoner_study, width_study
    t = time.time()
    for task in ("suma", "expr"):
        reasoner_study(task, f"{OUT}/reasoner_v2", L=8, steps=3000, per_gpu=2, base=dict(rope=False))
        print(f"== {task} terminado en {(time.time()-t)/60:.1f} min", flush=True)
    width_study("suma", f"{OUT}/reasoner_v2", base=dict(rope=False, L=8, steps=3000))
    print(f"== ancho terminado en {(time.time()-t)/60:.1f} min", flush=True)
