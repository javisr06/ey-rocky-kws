# Ey Rocky — prototipo local de wake word

Este directorio contiene solo el flujo de entrenamiento y los artefactos para integrarlos en una app Android ya existente. No crea una app y no envía audio a ningún servicio: el entrenamiento corre en este PC y el archivo `.tflite` se ejecuta en el teléfono.

## Datos de partida

`train.py` toma por defecto los audios de:

`C:\Users\jvrsn\OneDrive\Escritorio\Proyectos\audio`

Los clips cortos se consideran positivos; el archivo de más de 10 segundos se considera ruido de fondo. Los duplicados exactos se descartan. Confirma que los clips cortos dicen realmente «ey rocky» y que el largo es solo ruido antes de entrenar.

## Entrenar

Requisitos: Python 3.11–3.13 y conexión a Internet solo para instalar paquetes la primera vez. El procesamiento y entrenamiento de los audios es local.

```powershell
cd 'C:\Users\jvrsn\OneDrive\Escritorio\Proyectos\ey rocky'
python -m pip install -r requirements.txt
python train.py
```

También se puede indicar otra carpeta y más épocas:

```powershell
python train.py --audio-dir 'D:\mis-audios' --epochs 80
```

Salida en `artifacts\`:

- `ey_rocky.tflite`: modelo float32 optimizado para TensorFlow Lite.
- `ey_rocky.keras`: modelo editable para volver a entrenar.
- `training_report.json`: métricas diagnósticas y umbrales candidatos.

El informe incluye `provisional_threshold_from_small_validation_set`. Es un
umbral inicial elegido con seis positivos reservados y ventanas del único audio
de ruido; no lo tomes como un umbral validado para conversaciones reales.

## Integración Android

Añade el modelo a `app/src/main/assets/ey_rocky.tflite` y agrega TensorFlow Lite a la app. El modelo recibe 1 segundo de PCM mono a 16 kHz, float32 normalizado a `[-1, 1]`, con forma `[1, 16000, 1]`. La salida es una probabilidad entre 0 y 1. Usa una ventana deslizante y exige varias puntuaciones sobre el umbral antes de activar.

```kotlin
val interpreter = Interpreter(loadModelFile(context, "ey_rocky.tflite"))
val input = Array(1) { Array(16_000) { FloatArray(1) } }
for (i in 0 until 16_000) input[0][i][0] = pcm16k[i].coerceIn(-1f, 1f)
val output = Array(1) { FloatArray(1) }
interpreter.run(input, output)
val score = output[0][0]
```

## Limitaciones

Esto es un prototipo, no una garantía de detección robusta. Hay unos 30 positivos únicos y una sola grabación de ruido. La augmentación de volumen/velocidad y la mezcla con ruido no sustituyen conversaciones reales como ejemplos negativos. El informe mide el ruido disponible, no falsas activaciones en habla. Ajusta el umbral tras probarlo en el teléfono y conserva los fallos para reentrenar después.
