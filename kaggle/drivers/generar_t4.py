# NAVROS — generación de prueba con pesos publicados como release de GitHub (T4, fp32).
# Descarga los pesos, verifica su SHA-256, comprueba la caché KV contra el forward completo
# y continúa un conjunto fijo de frases en español e inglés (results/generacion/paso_XXXXX.md).
#@@HEADER@@
TAG = "@@TAG@@"
SHA256 = "@@SHA@@"
URL = f"https://github.com/navroscol/navros-ai/releases/download/{TAG}/pesos_bf16.pt"
if __name__ == "__main__":
    import hashlib
    env_report()
    dst = "/tmp/pesos_bf16.pt"  # fuera de /kaggle/working: no debe acabar en la salida del notebook
    t0 = time.time()
    subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-C", "-", "-o", dst, URL], check=True)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    print(f"descargado {os.path.getsize(dst):,} bytes en {time.time() - t0:.0f}s; sha256 {h.hexdigest()}", flush=True)
    assert h.hexdigest() == SHA256, "SHA-256 distinto del publicado"
    subprocess.run([sys.executable, f"{ROOT}/scripts/07_generar.py", "--pesos", dst, "--dtype", "fp32",
                    "--out", f"{OUT}/generacion"], check=True)
