# NAVROS — corpus bilingüe pretokenizado (CPU, no consume cuota de GPU/TPU).
# Partición 0/4: entrena el tokenizador BPE (32768) desde cero, verifica ida y vuelta,
# crea val/test por idioma y hasta 4B tokens de entrenamiento por idioma (≤ 11 h).
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.dataprep import prepare
    prepare("/kaggle/working/navros_data", target_tokens_per_lang=4_000_000_000, time_budget_s=11 * 3600, part=(0, 4),
            log=lambda s: print(s, flush=True))
