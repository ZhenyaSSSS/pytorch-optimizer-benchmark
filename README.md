# New_SAM: A²SAM & HATAM Optimizers Benchmark

## 🔬 Происхождение Проекта: Эксперимент в AI-Driven Research

Этот проект представляет собой нечто большее, чем просто бенчмарк. Он является результатом эксперимента по оценке возможностей **сложной мультиагентной системы на базе Google Gemini 2.5 Pro** в создании новых научных концепций.

Ключевая цель — проверить, способна ли современная AI-система с практически неограниченным бюджетом на вычисления (в рамках данного исследования было использовано ~400,000 токенов) провести глубокий синтез существующих знаний и предложить **новаторские научные идеи**.

**Центральными артефактами этого исследования являются сами научные труды (`A²SAM.md`, `HATAM.md`)**, которые были сгенерированы в результате работы системы. Код в этом репозитории — это уже вторичный продукт, практическая реализация предложенных теорий, созданная для их валидации и содержащая значительную долю человеческого участия в разработке и отладке.

Реализация и честное сравнение передовых оптимизаторов **A²SAM** (Accelerated Anisotropic Sharpness-Aware Minimization) и **HATAM** (Heuristic Anisotropic Trajectory-Aware Minimization) с базовыми методами на CIFAR-10.

## 🔬 Ключевые возможности

- **Корректные реализации** алгоритмов согласно оригинальным статьям
- **Честное сравнение** с одинаковыми условиями для всех оптимизаторов  
- **Robustness метрики** через CIFAR-10-C
- **Generalization gap** для оценки способности к обобщению
- **Детальные логи** времени оптимизации и производительности

## 🚀 Быстрый старт

### 1. Базовое сравнение
```bash
# Сравнение основных оптимизаторов
python train.py --optim hatam --epochs 30
python train.py --optim a2sam --epochs 30
python train.py --optim adam --epochs 30
```

### 2. С tracking обобщения
```bash
# Детальное отслеживание generalization gap
python train.py --optim hatam --epochs 30 --track-generalization
```

### 3. С robustness оценкой (скачает CIFAR-10-C 2.9GB)
```bash
# Полная оценка включая robustness
python train.py --optim hatam --epochs 30 --eval-robustness --track-generalization
```

### 4. Комплексное сравнение всех оптимизаторов
```bash
# Автоматическое сравнение с визуализацией
python compare_optimizers.py --optimizers adam hatam a2sam --epochs 20
python compare_optimizers.py --optimizers adam hatam --eval-robustness
```

### 5. Быстрый тест robustness (без скачивания)
```bash
# Синтетические коррупции для быстрой проверки
python test_robustness_synthetic.py
```

## 📊 Метрики оценки

### Основные метрики:
- **Test Accuracy** - точность на тестовых данных (чем выше, тем лучше)
- **Generalization Gap** - разность train_acc - test_acc (чем ниже, тем лучше)
- **Training Time** - время обучения (чем меньше, тем лучше)

### Robustness метрики:
- **mCE (mean Corruption Error)** - средняя ошибка на коррупциях, нормализованная по AlexNet (чем ниже, тем лучше)
- **Individual corruption errors** - ошибки по отдельным типам коррупций

### Интерпретация:
- ✅ **Generalization gap < 5%** - отличная способность к обобщению
- ✅ **mCE < 60%** - отличная robustness
- ⚠️ **Generalization gap 5-10%** - хорошее обобщение  
- ⚠️ **mCE 60-80%** - приемлемая robustness

## 🔬 Алгоритмы

### A²SAM (Accelerated Anisotropic SAM)
- **Анизотропное возмущение** вместо изотропного SAM
- **Аппроксимация гессиана** через power iteration
- **Амортизация вычислений** - обновление каждые M шагов
- **Формула Woodbury** для эффективного обращения матрицы

### HATAM (Heuristic Anisotropic Trajectory-Aware Minimization)  
- **Анизотропное выпрямление траектории** через разность градиентов
- **EMA сглаживание**: `c_t = β_c·c_{t-1} + (1-β_c)·(g_t - g_{t-1})`
- **Модификация градиента**: `g_hatam = g + γ·S⊙c`
- **1× вычислительная стоимость** (как Adam)

## 📁 Структура проекта

```
New_SAM/
├── optimizers/
│   ├── a2sam.py      # A²SAM реализация
│   └── hatam.py      # HATAM реализация
├── models/
│   ├── convnet.py    # Small ConvNet для CIFAR-10
│   └── mlp_mixer.py  # MLP-Mixer архитектура
├── utils/
│   ├── cifar10_c.py  # CIFAR-10-C загрузка и оценка
│   └── seed.py       # Воспроизводимость
├── tests/            # Unit тесты
├── train.py          # Основной скрипт обучения
├── compare_optimizers.py  # Комплексное сравнение
└── test_robustness_synthetic.py  # Быстрый robustness тест
```

## 🔧 Установка зависимостей

```bash
pip install -r requirements.txt
```

### Зависимости:
- torch >= 1.13
- torchvision >= 0.14  
- numpy
- matplotlib
- requests (для CIFAR-10-C)
- torch-optimizer (для SAM baseline)

## 💾 CIFAR-10-C Dataset

### Автоматическая загрузка:
```bash
# Датасет автоматически скачается при использовании --eval-robustness
python train.py --eval-robustness --optim hatam
```

### Ручная загрузка:
- **Источник**: https://zenodo.org/records/2535967  
- **Размер**: 2.9 GB
- **Содержание**: 15 типов коррупций × 5 уровней severity × 10,000 изображений

### Альтернатива - синтетические коррупции:
```bash
# Если не хотите скачивать полный датасет
python test_robustness_synthetic.py
```

## 📈 Примеры результатов

### Типичные результаты (30 epochs):
```
Optimizer  Accuracy  Gen Gap  mCE    Time   Score
------------------------------------------------------
HATAM      85.2%     3.1%     68.4%  180s   82.1
A2SAM      84.8%     2.8%     65.2%  220s   82.0  
ADAM       82.1%     8.5%     78.9%  160s   73.6
SAM        84.5%     3.2%     66.8%  350s   81.3
```

### Файлы результатов:
- `training_results_{optimizer}_{model}_seed{seed}.json` - подробная история
- `optimizer_comparison.png` - визуальное сравнение

## 🧪 Тестирование

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Тест только оптимизаторов
python -m pytest tests/test_optimizers.py -v

# Smoke test с fake data
python train.py --fake-data --epochs 1 --optim hatam
```

## 🏆 Лучшие практики

### Для исследований:
```bash
# Полная оценка с несколькими seed'ами
for seed in 42 123 456; do
    python train.py --optim hatam --seed $seed --epochs 50 \
                   --eval-robustness --track-generalization
done
```

### Для разработки:
```bash
# Быстрый тест новых оптимизаторов
python compare_optimizers.py --optimizers adam new_optimizer \
                            --epochs 5 --fake-data
```

### Для production:
```bash
# Максимальная точность
python train.py --optim hatam --epochs 200 --lr 1e-3 \
               --eval-robustness --track-generalization
```

## 📚 Научные статьи

Реализации основаны на теоретических концепциях, описанных в:
- `A²SAM.txt` - Accelerated Anisotropic Sharpness-Aware Minimization
- `HATAM.txt` - Heuristic Anisotropic Trajectory-Aware Minimization

## 🤝 Контрибьюции

Проект следует принципам современного программирования:
- ✅ Воспроизводимость через фиксированные seeds
- ✅ Модульная архитектура 
- ✅ Качественное логирование
- ✅ Покрытие тестами
- ✅ Честное сравнение методов

## 📄 Лицензия

Проект распространяется для исследовательских целей. 