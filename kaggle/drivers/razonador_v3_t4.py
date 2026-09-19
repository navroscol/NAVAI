# NAVROS — razonador v3: expresiones con RoPE y POSICIONES ALEATORIZADAS (Ruoss et al., 2023):
# en entrenamiento y prueba, las posiciones son un subconjunto ordenado al azar de [0, 256), así el
# modelo ve distancias relativas largas sin ver secuencias largas. Mismo protocolo que v1/v2. GPU T4 x2.
#@@HEADER@@
if __name__ == "__main__":
    env_report()
    from navros.experiments import reasoner_study
    t = time.time()
    reasoner_study("expr", f"{OUT}/reasoner_v3", L=8, steps=3000, lrs=(0.005, 0.01), per_gpu=2,
                   base=dict(rope=True, rope_range=256))
    print(f"== expr terminado en {(time.time()-t)/60:.1f} min", flush=True)
