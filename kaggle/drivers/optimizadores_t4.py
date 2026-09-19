# NAVROS — Muon contra AdamW, mejor contra mejor (GPU T4 x2).
# LM a nivel de bytes sobre FineWeb-Edu; anchos 128/256/512; horizontes 250 y 1000 pasos; 3 semillas.
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.data import prepare_text_corpus
    from navros.lm_sweep import optimizer_study
    corpus = prepare_text_corpus("/kaggle/working/data/fineweb_edu_64MB.npz")
    print(open(str(corpus).replace(".npz", ".json")).read(), flush=True)
    t = time.time()
    optimizer_study(f"{OUT}/optimizadores", corpus, per_gpu=2)
    print(f"== terminado en {(time.time()-t)/60:.1f} min", flush=True)
