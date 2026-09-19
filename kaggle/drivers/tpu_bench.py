# NAVROS — TPU v5e-8: sonda del entorno + benchmark del camino XLA con tokens sintéticos.
# Ejecutar desde el editor: Accelerator = TPU v5e-8, Internet = ON, Save Version → Save & Run All (Commit).
# Pasos: 1) entorno  2) prueba mínima (aborta si XLA no funciona)  3) NAVROS-1B recurrente y la pila fija de 1B.
#@@HEADER@@
if __name__ == "__main__":
    import glob, platform, traceback
    info = dict(python=sys.version, platform=platform.platform(), cpus=os.cpu_count(),
                accel=sorted(glob.glob("/dev/accel*")) + sorted(glob.glob("/dev/vfio/*")))
    pip = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True).stdout
    info["pkgs"] = [l for l in pip.splitlines() if any(s in l.lower() for s in ("torch", "jax", "xla", "libtpu"))]
    try:
        import torch_xla, torch_xla.runtime as xr
        info["torch_xla"] = torch_xla.__version__
        info["n_dev"] = xr.global_runtime_device_count()
    except Exception as e:
        info["torch_xla_error"] = repr(e)[:400]
    print(json.dumps(info, indent=1), flush=True)
    json.dump(info, open(f"{OUT}/sonda_tpu.json", "w"), indent=1)
    if "torch_xla_error" in info:
        raise SystemExit("sin torch_xla: nada que medir")
    from navros.lm import benchmark
    results = []
    for kw in [dict(preset="tiny", T=256, micro=8, steps=3, warmup=2),
               dict(preset="navros-1b", T=2048, micro=16, steps=8, warmup=3, grad_ckpt=True),
               dict(preset="navros-1b-fix", T=2048, micro=16, steps=8, warmup=3, grad_ckpt=True)]:
        try:
            results.append(benchmark(device="xla", precision="bf16", log=lambda s: print(s, flush=True), **kw))
        except Exception:
            results.append(dict(kw, error=traceback.format_exc()[-2000:]))
            print(results[-1]["error"], flush=True)
            if kw["preset"] == "tiny":
                break
        json.dump(results, open(f"{OUT}/tpu_bench.json", "w"), indent=1, default=str)
