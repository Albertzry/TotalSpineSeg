

# SpineSegPlus

> **Nota:** Este proyecto se basa en [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) desarrollado por [NeuroPoly Lab](https://neuro.polymtl.ca/). Hemos realizado extensiones arquitectónicas significativas sobre el proyecto original, que incluyen un flujo de trabajo (pipeline) de segmentación en dos etapas para la degeneración del disco intervertebral lumbar (degeneración IVD), un módulo integral de cálculo de parámetros clínicos y un flujo de inferencia de extremo a extremo. Consulte la sección [Agradecimientos](#acknowledgments) para la cita original.

---

## Tabla de Contenidos

- [Descripción General](#overview)
- [Características Principales](#key-features)
- [Arquitectura](#architecture)
  - [Etapa 1 — Segmentación gruesa y puntos de referencia (Dataset 101)](#stage-1--coarse-segmentation--landmarks-dataset-101)
  - [Etapa 2 — Etiquetado de gran detalle (Dataset 102)](#stage-2--fine-grained-labeling-dataset-102)
  - [Etapa 3 — Detección y segmentación en dos etapas de degeneración IVD (Dataset 105)](#stage-3--ivd-degeneration-two-stage-detection--segmentation-dataset-105)
  - [Cálculo de parámetros clínicos](#clinical-parameter-computation)
- [Estructura del Proyecto](#project-structure)
- [Dependencias](#dependencies)
- [Instalación](#installation)
- [Inferencia](#inference)
  - [Segmentación de la columna (Paso 1 y Paso 2)](#spine-segmentation-step-1--step-2)
  - [Inferencia de degeneración IVD (Pipeline de dos etapas)](#ivd-degeneration-inference-two-stage-pipeline)
  - [Generación de informes clínicos](#clinical-report-generation)
- [Entrenamiento](#training)
  - [Entrenamiento de los Pasos 1 y 2](#step-1--step-2-training)
  - [Entrenamiento en dos etapas de degeneración IVD (Dataset 105)](#ivd-degeneration-two-stage-training-dataset-105)
- [Estructura de datos de salida](#output-data-structure)
- [Lista de clases](#list-of-classes)
- [Documentación](#documentation)
- [Agradecimientos](#acknowledgments)
- [Licencia](#license)

---

## Descripción General

**SpineSegPlus** es una herramienta integral para el análisis automático de imágenes de resonancia magnética (RM) de la columna vertebral. Basándose en el [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) original, este proyecto extiende el pipeline con:

1. **Segmentación por instancia de toda la columna** — Segmentación y etiquetado automáticos de todas las vértebras (C1–L5), discos intervertebrales (IVD), sacro, médula espinal y canal vertebral, robusta ante diversos contrastes de RM, orientaciones de adquisición y resoluciones.
2. **Detección y segmentación en dos etapas de degeneración IVD** — Un pipeline novedoso que detecta y segmenta la degeneración del disco intervertebral lumbar utilizando conocimientos previos anatómicos y funciones de pérdida sensibles a la superficie.
3. **Cálculo de mediciones clínicas** — Cálculo automatizado de parámetros espinales clínicamente relevantes (altura vertebral, altura discal, ángulos de Cobb, lordosis lumbar, pendiente sacra, ángulo lumbosacro, ángulo de inclinación discal, métricas de protrusión por degeneración IVD, etc.) con visualización.

La arquitectura base (backbone) del modelo se basa en [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

---

## Características Principales

| Característica | Descripción |
|:--------|:------------|
| **Segmentación de columna en múltiples etapas** | Paso 1 (segmentación gruesa de 9 clases + detección de puntos de referencia) → Paso 2 (etiquetado detallado de 11 clases con vértebras impares/par) |
| **Pipeline de dos etapas para degeneración IVD** | Etapa A (detección binaria a nivel de disco usando 3D ResNet con atención SE) → Etapa B (segmentación fina basada en ROI usando 3D U-Net ligero con supervisión SDM) |
| **Inyección de conocimiento previo anatómico** | Mapas de índice de disco, atención en el borde del disco, atención en la interfaz disco-canal y atención en el espacio intervertebral |
| **Informe clínico integral** | Informe basado en JSON con alturas vertebrales, alturas discales, HDR/DHI, ángulos de alineación sagital (LL, SS, LSA, DIA), parámetros de degeneración IVD (PD, PA, PAR, PLR) e intensidad de señal del disco |
| **Visualización** | Generación automática de imágenes de vista previa de mediciones con superposiciones anotadas |
| **Soporte para localizador** | Etiquetado mejorado para imágenes con FOV limitado usando referencia basada en localizador |
| **Robustez multicontraste** | Validada en imágenes T1w, T2w, STIR, MTS, T2* e incluso CT |

---

## Arquitectura

### Etapa 1 — Segmentación gruesa y puntos de referencia (Dataset 101)

- **Entrada**: RM de canal único (remuestreada a 1mm isótropo, reorientada a LPI)
- **Salida**: 9 clases — médula espinal, canal vertebral, IVD, vértebras y 5 clases de puntos de referencia (discos clave C2–C3, C7–T1, T12–L1, L5–S y vértebra C1)
- **Postprocesamiento**: El algoritmo de etiquetado iterativo asigna índices anatómicos; los IVD con número impar se extraen para el Paso 2

### Etapa 2 — Etiquetado de gran detalle (Dataset 102)

- **Entrada**: 2 canales — imagen RM + máscara de IVD impar de la Etapa 1
- **Salida**: 11 clases — médula espinal, canal vertebral, IVD, vértebras impares/par, sacro y 4 discos de punto de referencia
- **Postprocesamiento**: El etiquetado iterativo reconstruye etiquetas C1–L5 + sacro; el llenado del canal garantiza la continuidad anatómica

### Etapa 3 — Detección y segmentación en dos etapas de degeneración IVD (Dataset 105)

Un pipeline novedoso de dos etapas diseñado específicamente para el desequilibrio extremo de clases y el pequeño volumen de lesiones de la degeneración del disco intervertebral lumbar:

```
┌──────────────────────────────────────────────────────────────────┐
│  Etapa A — Detección (Clasificación binaria a nivel de disco)    │
│  • Entrada: Parche de RM (96³) + disc_mask + disc_index_map     │
│  • Modelo: StageADetectorV2 (CNN 3D residual + atención SE)      │
│  • Pérdida: Focal Loss (γ=2)                                     │
│  • Salida: has_IVD (0 o 1) por nivel de disco                    │
└──────────────────────────────────────┬───────────────────────────┘
                                       ↓ (solo discos positivos)
┌──────────────────────────────────────────────────────────────────┐
│  Etapa B — Segmentación fina (Predicción de máscara IVD basada en│
│                      ROI)                                        │
│  • Entrada: ROI de RM (48³) + disc_mask + disc_index_map        │
│  • Modelo: SmallUNet3D (3D U-Net de 3 niveles, doble cabeza)    │
│  • Pérdida: FocalTversky + Boundary + L1(SDM)                   │
│  • Salida: Máscara binaria de degeneración IVD + mapa de         │
│            distancia con signo (SDM)                             │
└──────────────────────────────────────────────────────────────────┘
```

**Innovaciones clave:**
- **Mapas de índice de disco**: Codificación posicional normalizada (0–1) derivada de las etiquetas de vértebra del Paso 2, que proporciona conocimientos previos espaciales
- **Muestreo obligatorio de 4 clases**: Cada disco genera 4 tipos de parche (centro de degeneración IVD, borde de degeneración IVD, negativo en el borde del disco, negativo en el interior del disco) para garantizar un entrenamiento equilibrado
- **Supervisión sensible a la superficie**: La regresión del mapa de distancia con signo (SDM) obliga a la red a aprender la geometría de los bordes
- **Atención anatómica**: Mapas de atención multinivel (borde del disco, interfaz disco-canal, espacio intervertebral) restringen las predicciones a regiones anatómicamente plausibles

### Cálculo de parámetros clínicos

El módulo `calculate.py` proporciona el cálculo automatizado de mediciones espinales clínicamente relevantes:

| Categoría | Parámetros |
|:---------|:-----------|
| **Morfometría vertebral** | Altura vertebral anterior/posterior (VH), diámetro AP del cuerpo vertebral |
| **Morfometría discal** | Altura discal (DH) en ubicaciones anterior/media/posterior, relación altura-disco (HDR), índice de altura discal (DHI) |
| **Alineación sagital** | Lordosis lumbar (LL), Pendiente sacra (SS), Ángulo lumbosacro (LSA) |
| **Ángulos discales** | Ángulo de inclinación discal (DIA) por nivel |
| **Parámetros de degeneración IVD** | Distancia de protrusión (PD), Área de protrusión (PA), Relación de PA (PAR), Relación de protrusión-longitud (PLR) |
| **Análisis de señal** | Nivel de gris promedio (AGL) por disco |

Todas las mediciones se generan como un informe JSON estructurado con imágenes de visualización acompañantes.

---

## Estructura del Proyecto

```
SpineSegPlus/
├── totalspineseg/                  # Paquete principal
│   ├── __init__.py                 # Exportaciones del paquete
│   ├── inference.py                # Inferencia principal de segmentación de columna (Paso 1 y Paso 2)
│   ├── init_inference.py           # Inicialización del modelo y descarga de pesos
│   ├── ldh_twostage/               # ★ NUEVO: Módulo de pipeline de dos etapas para degeneración IVD
│   │   ├── models.py               #   Arquitecturas StageADetectorV2 + SmallUNet3D
│   │   ├── losses.py               #   Pérdidas FocalTversky, Boundary, SDM
│   │   ├── sampling.py             #   Muestreo obligatorio de parches de 4 clases
│   │   ├── disc_index.py           #   Generación de mapas de índice de disco a partir de etiquetas del Paso 2
│   │   ├── distance_maps.py        #   Cálculo de mapas de distancia con signo
│   │   ├── data.py                 #   Clases Dataset de PyTorch
│   │   └── metrics.py              #   Métricas de evaluación
│   ├── nnunet_extensions/          # Extensiones del entrenador nnU-Net
│   ├── resources/                  # Mapas de etiquetas y configuraciones de conjuntos de datos
│   │   ├── labels_maps/            #   tss_map.json, nnunet_step1/2/5_ldh.json, etc.
│   │   └── datasets/              #   Configuraciones de conjuntos de datos
│   └── utils/                      # Módulos de utilidad
│       ├── iterative_label.py      #   Algoritmo de etiquetado anatómico iterativo
│       ├── predict_nnunet.py       #   Wrapper de predicción nnU-Net (con monkeypatch)
│       ├── extract_alternate.py    #   Extracción de discos impares/par
│       ├── extract_levels.py       #   Extracción de niveles de disco
│       ├── fill_canal.py           #   Reparación de topología del canal
│       ├── resample.py             #   Remuestreo de imágenes
│       └── ...                     #   Otras utilidades
│
├── calculate.py                    # ★ NUEVO: Cálculo de parámetros clínicos y generación de informes
├── example_usage.py                # Ejemplos de uso para integración programática
│
├── scripts/                        # Scripts de entrenamiento e inferencia
│   ├── prepare_dataset_105.py      # ★ NUEVO: Preparación de datos del Dataset 105 (degeneración IVD)
│   ├── train_ldh_stage_a.py        # ★ NUEVO: Entrenamiento de la Etapa A de degeneración IVD
│   ├── train_ldh_stage_b.py        # ★ NUEVO: Entrenamiento de la Etapa B de degeneración IVD
│   ├── infer_ldh.py                # ★ NUEVO: Pipeline de inferencia de extremo a extremo para degeneración IVD
│   ├── eval_ldh.py                 # ★ NUEVO: Evaluación de degeneración IVD
│   ├── prepare_datasets.sh         #   Preparación de conjuntos de datos 101/102/103
│   ├── download_datasets.sh        #   Descarga de conjuntos de datos
│   └── train.sh                    #   Punto de entrada unificado de entrenamiento
│
├── docs/                           # Documentación
│   ├── LDH_TwoStage_Pipeline.md   #   Documentación detallada del pipeline de degeneración IVD
│   ├── LDH_Quick_Reference_CN.md  #   Referencia rápida para degeneración IVD (Chino)
│   ├── Step1_2_5_Technical_Report_bilingual.md  # Informe técnico
│   └── ...                         #   Otra documentación
│
├── pyproject.toml                  # Configuración del paquete
├── LICENSE                         # Archivo de licencia
└── README.md                       # Este archivo
```

---

## Dependencias

- **Python** >= 3.10, con pip >= 23 y setuptools >= 67
- **PyTorch** < 2.6 (se recomienda soporte CUDA)
- **nnU-Net v2** (arquitectura base para entrenamiento/inferencia del Paso 1 y Paso 2)
- Bibliotecas clave: `nibabel`, `SimpleITK`, `nilearn`, `scipy`, `torchio`, `gryds`, `tqdm`, `matplotlib`

---

## Instalación

1. Cree y active un entorno virtual:
   ```bash
   conda create -n tss python=3.10
   conda activate tss
   ```

2. Instale SpineSegPlus:
   ```bash
   git clone <repository-url> SpineSegPlus
   cd SpineSegPlus
   python3 -m pip install -e .[nnunetv2]
   ```

3. Instale PyTorch con soporte CUDA:
   ```bash
   python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu118 --upgrade
   ```

4. (Opcional) Establezca el directorio de datos del modelo:
   ```bash
   export TOTALSPINESEG_DATA="/path/to/data"
   ```

---

## Inferencia

### Segmentación de la columna (Paso 1 y Paso 2)

```bash
# Procesar un archivo o carpeta NIfTI individual
totalspineseg INPUT OUTPUT_FOLDER [--step1] [--iso]

# Ejemplos
totalspineseg input.nii.gz output_folder
totalspineseg input_folder output_folder --iso --device cuda

# Con localizador para imágenes con FOV limitado
totalspineseg localizers localizers_output --iso
totalspineseg images output --loc localizers_output/step2_output --suffix _T2w --loc-suffix _T1w
```

Para ver todas las opciones, ejecute `totalspineseg --help`.

### Inferencia de degeneración IVD (Pipeline de dos etapas)

```bash
python scripts/infer_ldh.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output \
    --device cuda
```

El pipeline de inferencia de degeneración IVD realizará:
1. Ejecutar el Paso 1 + Paso 2 para obtener la segmentación anatómica completa
2. Generar mapas de índice de disco a partir de las etiquetas del Paso 2
3. Para cada nivel de disco lumbar (L1/L2 a L5/S1):
   - **Etapa A**: Detectar si hay degeneración IVD presente
   - **Etapa B** (si es positivo): Segmentar la región de degeneración IVD en una ROI enfocada
4. Agregar todas las predicciones a nivel de disco en una máscara final de degeneración IVD

### Generación de informes clínicos

```bash
python calculate.py \
    --input-dir /path/to/processed_cases \
    --output-dir /path/to/reports
```

Esto genera:
- **Informe JSON** con todas las mediciones clínicas (altura vertebral, altura discal, ángulos de Cobb, parámetros de degeneración IVD, etc.)
- **Imágenes de visualización** con superposiciones anotadas de mediciones para cada parámetro

---

## Entrenamiento

### Entrenamiento de los Pasos 1 y 2

**Requisitos de hardware:**
- ~3.5 TB de espacio en disco (con aumento de datos)
- ≥32 GB de RAM
- GPU CUDA con ≥8 GB de VRAM

```bash
# Establecer variables de entorno
export TOTALSPINESEG="$(realpath .)"
export TOTALSPINESEG_DATA="/path/to/data"

# Descargar conjuntos de datos
bash scripts/download_datasets.sh

# Preparar conjuntos de datos nnUNet
bash scripts/prepare_datasets.sh [101|102|103|all] [-noaug]

# Entrenar
bash scripts/train.sh [DATASET_ID [FOLD]]
```

### Entrenamiento en dos etapas de degeneración IVD (Dataset 105)

```bash
# 1. Preparar Dataset 105 (genera parches de Etapa A + ROIs de Etapa B)
python scripts/prepare_dataset_105.py \
    --stagea-patch 96 \
    --stageb-roi 48 \
    --device cuda

# 2. Entrenar (un solo comando ejecuta ambas etapas + evaluación)
bash scripts/train.sh 105 0

# O ejecutar etapas individualmente:
python scripts/train_ldh_stage_a.py --epochs 50 --batch-size 32 --lr 1e-3
python scripts/train_ldh_stage_b.py --epochs 200 --batch-size 16 --lr 1e-4

# 3. Evaluar
python scripts/eval_ldh.py \
    --ckpt-dir /path/to/checkpoints \
    --device cuda
```

**Estimaciones de tiempo de entrenamiento (V100/A100 individual):**
| Etapa | Épocas | Tiempo |
|:------|:-------|:-----|
| Etapa A (Detección) | 50 | ~2–4 horas |
| Etapa B (Segmentación) | 200 | ~8–12 horas |

---

## Estructura de datos de salida

### Salida de segmentación de columna
```
output_folder/
├── input/              # Imágenes de entrada preprocesadas (1mm iso, LPI)
├── preview/            # Imágenes de vista previa (JPEG)
├── step1_raw/          # Salida cruda del modelo del Paso 1
├── step1_output/       # Resultado de etiquetado iterativo del Paso 1
├── step1_cord/         # Segmentación suave de la médula espinal
├── step1_canal/        # Segmentación suave del canal vertebral
├── step1_levels/       # Marcadores de nivel de disco de un solo voxel en la línea central del canal
├── step2_raw/          # Salida cruda del modelo del Paso 2
└── step2_output/       # Segmentación etiquetada final (vértebras, discos, médula, canal)
```

### Salida de informes clínicos
```
report_output/
├── result/
│   ├── report.json                     # Informe completo de mediciones en JSON
│   └── previews/                       # Imágenes de visualización
│       ├── vertebrae/                  #   Medidas de altura y ancho vertebrales
│       ├── discs/                      #   Medidas de altura discal y DIA
│       └── global/                     #   Visualizaciones de ángulos LL, SS, LSA
└── raw/                                # Datos intermedios de cálculo
```

---

## Lista de clases

> El mapeo también está disponible en `totalspineseg/resources/labels_maps/tss_map.json`

| Etiqueta | Nombre |
|:------|:-----|
| 1 | spinal_cord |
| 2 | spinal_canal |
| 11 | vertebrae_C1 |
| 12 | vertebrae_C2 |
| 13 | vertebrae_C3 |
| 14 | vertebrae_C4 |
| 15 | vertebrae_C5 |
| 16 | vertebrae_C6 |
| 17 | vertebrae_C7 |
| 21 | vertebrae_T1 |
| 22 | vertebrae_T2 |
| 23 | vertebrae_T3 |
| 24 | vertebrae_T4 |
| 25 | vertebrae_T5 |
| 26 | vertebrae_T6 |
| 27 | vertebrae_T7 |
| 28 | vertebrae_T8 |
| 29 | vertebrae_T9 |
| 30 | vertebrae_T10 |
| 31 | vertebrae_T11 |
| 32 | vertebrae_T12 |
| 41 | vertebrae_L1 |
| 42 | vertebrae_L2 |
| 43 | vertebrae_L3 |
| 44 | vertebrae_L4 |
| 45 | vertebrae_L5 |
| 50 | sacrum |
| 63 | disc_C2_C3 |
| 64 | disc_C3_C4 |
| 65 | disc_C4_C5 |
| 66 | disc_C5_C6 |
| 67 | disc_C6_C7 |
| 71 | disc_C7_T1 |
| 72 | disc_T1_T2 |
| 73 | disc_T2_T3 |
| 74 | disc_T3_T4 |
| 75 | disc_T4_T5 |
| 76 | disc_T5_T6 |
| 77 | disc_T6_T7 |
| 78 | disc_T7_T8 |
| 79 | disc_T8_T9 |
| 80 | disc_T9_T10 |
| 81 | disc_T10_T11 |
| 82 | disc_T11_T12 |
| 91 | disc_T12_L1 |
| 92 | disc_L1_L2 |
| 93 | disc_L2_L3 |
| 94 | disc_L3_L4 |
| 95 | disc_L4_L5 |
| 100 | disc_L5_S |

---

## Documentación

La documentación detallada está disponible en el directorio `docs/`:

- **[LDH_TwoStage_Pipeline.md](docs/LDH_TwoStage_Pipeline.md)** — Arquitectura integral del pipeline de degeneración IVD, detalles de implementación y guía de uso
- **[LDH_Quick_Reference_CN.md](docs/LDH_Quick_Reference_CN.md)** — Referencia rápida para el pipeline de degeneración IVD (Chino)
- **[Step1_2_5_Technical_Report_bilingual.md](docs/Step1_2_5_Technical_Report_bilingual.md)** — Informe técnico bilingüe para el Paso 1, Paso 2 y Paso 5

---

## Agradecimientos

Este proyecto se basa y extiende el trabajo de [TotalSpineSeg](https://github.com/neuropoly/totalspineseg) desarrollado por el [NeuroPoly Lab](https://neuro.polymtl.ca/) en Polytechnique Montréal. Cite el trabajo original si utiliza este proyecto:

```bibtex
@article{warszawer2025totalspineseg,
   title={TotalSpineSeg: Robust Spine Segmentation with Landmark-Based Labeling in MRI},
   author={Warszawer, Yehuda and Molinier, Nathan and Valosek, Jan and Benveniste, Pierre-Louis and Bédard, Sandrine and Shirbint, Emanuel and Mohamed, Feroze and Tsagkas, Charidimos and Kolind, Shannon and Lynd, Larry and Oh, Jiwon and Prat, Alexandre and Tam, Roger and Traboulsee, Anthony and Patten, Scott and Lee, Lisa Eunyoung and Achiron, Anat and Cohen-Adad, Julien},
   year={2025},
   journal={ResearchGate preprint},
   url={https://www.researchgate.net/publication/389881289_TotalSpineSeg_Robust_Spine_Segmentation_with_Landmark-Based_Labeling_in_MRI}
}
```

Cite también nnU-Net, ya que la arquitectura base de la segmentación se basa en gran medida en él:

```bibtex
@article{isensee2021nnunet,
   title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
   author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
   journal={Nature methods},
   volume={18},
   number={2},
   pages={203--211},
   year={2021}
}
```

---

## Licencia

Consulte el archivo [LICENSE](LICENSE) para obtener más detalles.
