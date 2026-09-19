# NAVROS — razonador: bucle recurrente contra referencia fija de mismo cómputo (GPU T4 x2).
# Tareas: suma (acarreos) y expresiones mod 10. Entrena con L=8, evalúa en 8, 16, 24, 32.
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    # El port verificado también en esta GPU (CUDA float32 contra el oráculo NumPy)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/02_verify_port.py"), "--out", f"{OUT}/verify_port_gpu.json"], cwd=ROOT)
    from navros.experiments import reasoner_study
    t = time.time()
    for task in ("suma", "expr"):
        reasoner_study(task, f"{OUT}/reasoner", L=8, steps=3000, per_gpu=2)
        print(f"== {task} terminado en {(time.time()-t)/60:.1f} min", flush=True)
