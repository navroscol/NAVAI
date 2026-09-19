# NAVROS — LM pequeño (~40M): recurrente contra pila fija de mismo cómputo y de mismos parámetros (GPU T4 x2).
# Necesita como entrada la salida del notebook navros-datos-p0 (corpus pretokenizado).
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.experiments import lm_study
    from navros.lm import find_data_dirs
    dirs = find_data_dirs()
    print("datos:", dirs, flush=True)
    assert dirs, "adjunta navros-datos-p0 como entrada"
    t = time.time()
    lm_study(f"{OUT}/lm_estudio", dirs)
    print(f"== terminado en {(time.time()-t)/60:.1f} min", flush=True)
