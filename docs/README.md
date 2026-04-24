# 📚 Documentación Central — Guías de Estudio

> **Hub de navegación: Toda la documentación del programa AI/ML en un solo lugar**

______________________________________________________________________

## 🎯 Si estás empezando

**Lectura en orden recomendado (15-20 min total):**

1. 📋 [**RESUMEN-MODULOS.md**](RESUMEN-MODULOS.md) (5 min)

   - Qué aprenderás módulo por módulo
   - Overview rápido de los 12 módulos core + 6 avanzados

1. 🗺️ [**LEARNING-PATH.md**](LEARNING-PATH.md) (10 min)

   - Ruta completa de estudio por fases
   - Secuencia sugerida sin deadlines fijos

1. ⏰ [**STUDY-RHYTHM.md**](STUDY-RHYTHM.md) (5 min)

   - Cómo sostener un ritmo realista
   - Tips de consistencia y disciplina

1. ⚡ [**QUICK-START.md**](QUICK-START.md) (10 min + práctica)

   - Setup del entorno en 3 minutos
   - Ejecuta tu primer código de ML

1. 🚀 **¡Comienza!**

   ```bash
   cd ../modules/01-programming-math-for-ml/
   cat README.md
   ```

______________________________________________________________________

## 📖 Guías Completas

### 🎓 **Para Estudiantes**

| Documento                                    | Propósito                             | Cuándo leer                |
| -------------------------------------------- | ------------------------------------- | -------------------------- |
| [**LEARNING-PATH.md**](LEARNING-PATH.md)     | Roadmap completo por fases sin fechas | ⭐ Antes de empezar        |
| [**RESUMEN-MODULOS.md**](RESUMEN-MODULOS.md) | Qué aprenderás en cada módulo         | Planificando tu ruta       |
| [**QUICK-START.md**](QUICK-START.md)         | Setup rápido + primer ejemplo         | ⚡ Para empezar HOY        |
| [**STUDY-RHYTHM.md**](STUDY-RHYTHM.md)       | Cómo mantener consistencia            | Si no sabes cuánto dedicar |
| [**RESOURCES.md**](RESOURCES.md)             | Libros, cursos, papers recomendados   | Durante el estudio         |

### 🛠️ **Para Mantenedores**

| Documento                                                | Propósito                         | Audiencia                     |
| -------------------------------------------------------- | --------------------------------- | ----------------------------- |
| [**IMPLEMENTATION-STATUS.md**](IMPLEMENTATION-STATUS.md) | Estado 100% completo del proyecto | Verificar qué está disponible |
| [**PHASES-CREACION.md**](PHASES-CREACION.md)             | Fases de construcción del repo    | Contribuidores y creadores    |
| [**ROADMAP-ORIGINAL.md**](ROADMAP-ORIGINAL.md)           | Referencia histórica inicial      | Contexto del diseño original  |
| [**COVERAGE-GAP-MATRIX.md**](COVERAGE-GAP-MATRIX.md)     | Matriz P0/P1/P2 de brechas        | Planificación de ejecución    |
| [**EXAMPLE-BACKLOG-PILOT.md**](EXAMPLE-BACKLOG-PILOT.md) | Backlog piloto de ejemplos        | Ejecución Fase 2              |

______________________________________________________________________

## 🗂️ Estructura del Repositorio

```
training-ai/
├── 📄 README.md                    → Punto de entrada principal
│
├── 📁 docs/                        → Documentación centralizada (estás aquí)
│   ├── LEARNING-PATH.md           → Ruta de estudio completa
│   ├── RESUMEN-MODULOS.md         → Qué aprenderás
│   ├── STUDY-RHYTHM.md            → Ritmo sugerido
│   ├── RESOURCES.md               → Recursos externos
│   ├── IMPLEMENTATION-STATUS.md   → Estado del proyecto
│   ├── PHASES-CREACION.md         → Fases de construcción
│   └── ROADMAP-ORIGINAL.md        → Referencia histórica
│
├── 📁 modules/                     → 12 módulos core (tu foco)
│   ├── 01-programming-math-for-ml/
│   ├── 02-data-collection-cleaning-visualization/
│   ├── ...
│   └── 12-mlops-ai-in-production/
│
├── 📁 trends-extras/               → 6 módulos avanzados
│   ├── agents/
│   ├── guardrails/
│   ├── multimodal/
│   ├── llm-evals/
│   ├── ai-observability/
│   └── synthetic-data/
│
├── 📁 shared/                      → Recursos compartidos
│   ├── datasets/                  → Datasets públicos
│   ├── notebooks/                 → Jupyter notebooks
│   └── utils/                     → Utilidades reutilizables
│
└── 📁 templates/                   → Plantillas reutilizables
    ├── PRACTICE.template.md
    ├── PROJECT.template.md
    └── EVALUATION-RUBRIC.template.md
```

______________________________________________________________________

## 🔗 Links Rápidos

### 🎯 **Empezar a Estudiar**

- [← Volver al README principal](../README.md)
- [→ Ir al Módulo 1](../modules/01-programming-math-for-ml/README.md)
- [📋 Ver resumen de todos los módulos](RESUMEN-MODULOS.md)

### 📊 **Estado del Proyecto**

- [✅ Estado de implementación](IMPLEMENTATION-STATUS.md) - 100% completo
- [📈 Fases de creación](PHASES-CREACION.md) - Cómo se construyó

### 📚 **Recursos Externos**

- [🔖 Recursos recomendados](RESOURCES.md) - Papers, cursos, libros

______________________________________________________________________

## 💡 Cómo navegar este repositorio

### 🎓 **Si quieres aprender AI/ML:**

```
1. Lee ../README.md (overview motivacional)
2. Lee RESUMEN-MODULOS.md (qué aprenderás)
3. Lee LEARNING-PATH.md (ruta completa)
4. Ve a modules/01-... y empieza
```

### 🔍 **Si buscas material específico:**

```
- Busca en trends-extras/ para temas avanzados
- Usa shared/datasets/ para datos de práctica
- Revisa templates/ para crear tus propios proyectos
```

### 🤝 **Si quieres contribuir:**

```
1. Lee IMPLEMENTATION-STATUS.md (qué está completo)
2. Lee PHASES-CREACION.md (arquitectura del repo)
3. Fork y abre un PR con mejoras
```

______________________________________________________________________

## 📝 Nota sobre `init-path/`

La carpeta `init-path/` contiene el primer draft del programa y se mantiene como referencia histórica.

**El contenido vigente y mantenido está en:**

- ✅ `modules/` - Módulos actualizados y completos
- ✅ `trends-extras/` - Tendencias actuales de la industria
- ✅ `docs/` - Documentación centralizada

______________________________________________________________________

## 🎯 Próximos Pasos

¿Listo para empezar? Sigue esta secuencia:

1. ✅ Leíste este archivo (docs/README.md) ← **Estás aquí**
1. → Lee [RESUMEN-MODULOS.md](RESUMEN-MODULOS.md) (5 min)
1. → Lee [LEARNING-PATH.md](LEARNING-PATH.md) (10 min)
1. → Abre [Módulo 1](../modules/01-programming-math-for-ml/README.md)
1. → ¡Empieza a programar! 💻

______________________________________________________________________

<div align="center">

**¿Dudas? ¿Sugerencias?**
Abre un issue en GitHub o inicia una Discussion

[← Volver al inicio](../README.md) | [Comenzar Módulo 1 →](../modules/01-programming-math-for-ml/README.md)

</div>
