## LM pequeño: recurrente contra pilas fijas (pérdida de prueba, nats/token)

| modelo | params | GFLOP/token entreno | LR Muon | test es | test en | tok/s |
|---|---|---|---|---|---|---|
| rec-s | 42,082,816 | 0.43 | 0.02 | 4.1230 | 4.3134 | 255.7K |
| fix20-s | 80,040,448 | 0.54 | 0.02 | 4.0193 | 4.1991 | 180.5K |
| fix8-s | 42,082,816 | 0.28 | 0.02 | 4.1342 | 4.3236 | 361.1K |

Pérdida de validación del recurrente según r (r̄ de entrenamiento = 4):

| idioma | r=1 | r=2 | r=3 | r=4 | r=6 | r=8 | r=12 | r=16 | adaptativo |
|---|---|---|---|---|---|---|---|---|---|
| es | 4.0263 | 4.0199 | 4.0195 | 4.0195 | 4.0196 | 4.0196 | 4.0196 | 4.0196 | 4.0195 (r̄=3.3) |
| en | 4.4552 | 4.4457 | 4.4449 | 4.4448 | 4.4447 | 4.4447 | 4.4447 | 4.4447 | 4.4447 (r̄=3.3) |
