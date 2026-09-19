# NAVROS — corpus bilingüe pretokenizado (CPU, no consume cuota de GPU/TPU).
# Partición 0/4 con el tokenizador del repo (navros/assets/tokenizer_navros_32k.json, el mismo que usa Modal):
# verifica ida y vuelta, crea val/test por idioma y TARGET tokens de entrenamiento por idioma.
#@@HEADER@@
TARGET = @@TARGET@@
if __name__ == "__main__":
    env_report()
    from navros.dataprep import prepare
    prepare("/kaggle/working/navros_data", target_tokens_per_lang=TARGET, time_budget_s=11 * 3600, part=(0, 4),
            tokenizer_path=os.path.join(ROOT, "navros/assets/tokenizer_navros_32k.json"), log=lambda s: print(s, flush=True))
    print("corpus terminado; salida inmediata (evita el cuelgue de los hilos de `datasets` al cerrar)", flush=True)
    sys.stdout.flush()
    os._exit(0)
