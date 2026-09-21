# Hacia dónde va todo: el "Pocket Frontier Model" y qué aporta cada pieza de este repositorio

Resumen del plan de implementación (documento de Pedro Álvarez, septiembre de 2026) y de cómo
encaja con lo construido aquí. Sirve para que cualquier trabajo nuevo se mida contra el objetivo.

## El objetivo, en una frase

Un asistente **offline** que corre en un teléfono Android o desde una USB con 3 a 6 GB de RAM y un
paquete de 4 a 8 GB, que conversa, programa, ejecuta, lee y modifica archivos, usa herramientas,
recuerda y **se corrige ejecutando y verificando**. Se mide por **tareas terminadas
correctamente** sobre una suite definida (100 a 200 pruebas), no por parámetros.

## La tesis del plan

La inteligencia deja de estar solo en los pesos. El sistema es:

    núcleo pequeño (2 a 4B, cuantizado a 1,58-4 bits)
    + router (~100M) + modelo de herramientas (~270M) + verificador (300-700M) + borrador (300-500M)
    + adaptadores LoRA por rol (código, matemáticas, razonamiento, shell, conversación, idiomas)
    + memoria episódica y RAG local
    + herramientas con permisos explícitos y sandbox
    + verificador que ejecuta (compilador, tests) y devuelve evidencia
    + bucle de agente: plan → acción → observación → corrección → resultado

Entrenamiento: trayectorias completas (tarea, pensamiento, llamada, observación, corrección,
respuesta), destiladas desde modelos frontera ("profesor gigante"), luego SFT de agente y RL con
recompensas por hechos (compila, pasan tests, termina, no destruye). Después compresión por
etapas (4 → 3 → 2 → 1,58 bits) midiendo la pérdida de capacidad en cada una. Tres perfiles:
Pocket (4 GB), Mobile Pro (6-8 GB), USB/PC (8-16 GB). Runtime propio (motor, cuantización, caché
KV, cargador de expertos, router, memoria, RAG, herramientas, sandbox, verificador, bucle).

Experimentos que deciden: A (precisión de bits), B (solo modelo / +RAG / +herramientas /
+verificador), C (con y sin destilación), D (adaptador de código), E (contexto comprimido),
F (un modelo contra router + especialistas). Prototipo V0: modelo de 2 a 4B a 2-4 bits + RAG +
memoria + herramientas + agente + verificador, y la pregunta que importa: **¿aguanta 20 a 50 pasos
sin intervención humana?**

## Qué encaja de lo hecho aquí, y qué no

| pieza de este repositorio | en el plan | valor |
|---|---|---|
| Phi-4-mini portado con tokenizador propio (`phi4mini`) | el **núcleo** de 3,8B, base MIT, multilingüe, ya sabe seguir instrucciones y razonar | alto: es el candidato a núcleo del V0 |
| trasplante OMP + cadena de tramos en Modal | cómo ese núcleo aprende nuestro corpus es/en sin retokenizar | medio: necesario si se conserva el tokenizador propio; el plan no lo exige |
| rejilla (`phi4mini-rejilla`: atención lineal con estado matricial) | **inferencia con memoria constante** en el contexto: en 4 GB de RAM la caché KV de la softmax es el cuello de botella para conversaciones y trayectorias largas | alto **si** iguala la perplejidad; es la pieza de este repositorio que más pega con "4 GB" |
| MPS sobre Collatz (superposición de estados de autómata) | la evidencia de que un estado recurrente pequeño ejecuta procedimientos exactos y generaliza en longitud; y la escala ordinal k como medida | medio: informa el diseño (estado recurrente para "procesos"), no es un componente |
| superposición de conceptos (rejilla ≈ vector bien ajustado) | descarta esperar más capacidad por geometría; la capacidad viene de herramientas, memoria y verificación, como dice el plan | alto como resultado negativo: ahorra tiempo |
| NAVROS-1B desde cero + 2.000 $ en tokens | un modelo propio y entendido de punta a punta; **no** es el núcleo del Pocket (le faltan instrucciones, herramientas y 3 a 10× de escala) | medio: donante del OMP, laboratorio de entrenamiento, marca; no invertir los 2.000 $ ahí si el objetivo es el Pocket |
| A/B de pérdida (LR, weight decay, contexto) | reglas para cualquier tramo de entrenamiento que hagamos | bajo-medio |
| proyección a 2B-40B | dice que el camino no es crecer: coincide con el plan | cierre |

Lo que el plan pide y aquí no existe todavía: el runtime, el sandbox y el protocolo de
herramientas; la memoria episódica y el RAG; el verificador; el dataset de trayectorias; la
cuantización (4 → 1,58 bits) y su medida; la suite de 100 a 200 tareas. Es la mayor parte del
trabajo, y es ingeniería de sistema, no entrenamiento.

## Consecuencias para lo que se decida a continuación

1. **El V0 no necesita entrenar nada.** Núcleo Phi-4-mini (o su port) cuantizado a 4 bits, con
   llama.cpp o un runtime propio, más herramientas, memoria y verificador. La primera medida es
   el experimento B (solo modelo / +RAG / +herramientas / +verificador) sobre 30 tareas propias.
   Eso se puede montar en semanas y no gasta GPU.
2. **La suite de tareas es lo primero que hay que escribir**, porque es la métrica de todo lo
   demás. Sin ella, cada pieza se optimiza a ciegas.
3. **La rejilla tiene aquí su justificación de producto**: memoria constante en el contexto para
   trayectorias de 20 a 50 pasos en 4 GB. La cadena B (`phi4mini-rejilla`) responde si cuesta
   perplejidad; si no cuesta, es un componente del Pocket, no un experimento.
4. **Los 2.000 $ de Modal** rinden más en destilación de trayectorias (profesor grande generando
   tareas, soluciones, errores y correcciones, y SFT de agente sobre el núcleo) que en
   preentrenar el 1B. El preentrenamiento continuado del núcleo con nuestro corpus (la cadena A)
   solo tiene sentido si el español de Phi-4-mini no basta; medirlo antes.
5. **NAVROS-1B sigue siendo el modelo propio** y el laboratorio; el Pocket es un sistema con un
   núcleo derivado (MIT) y hay que presentarlo así.
