# Chemotaxis-inspired decision memory

Контролируемое исследование: обгоняет ли память, организованная вокруг
**будущих решений** агента, память на основе **семантической близости**, и где это преимущество исчезает.

## Что смотреть

- [`report/REPORT.md`](report/REPORT.md) / [PDF](report/REPORT.pdf) -- русский
- [`report/REPORT_EN.md`](report/REPORT_EN.md) / [PDF](report/REPORT_EN.pdf) -- английский

Дополнительные ограничения и бэклог, не вошедшие в отчёт, приведены ниже.

## Структура репозитория

```
env/            синтетическая среда (правила, стили, генератор траекторий,
                controlled embedding для H2)
memory/         retrieval-ядро (rank_candidates) + голосующий слой
                (predict_action) + обучаемая альтернатива (learned.py)
agents/         Tier-1 kNN-vote агент, Tier-2 LLM-агент
realdata/       адаптеры LoCoMo и LongMemEval
experiments/    конфиг + все раннеры экспериментов
analysis/       графики, сводные таблицы, схема пайплайна, значимость
tests/          проверки чистой логики среды
results/        tables/ и plots/, всё, что цитирует отчёт
report/         статья
docker/         Dockerfile + docker-compose.yml
```

## Воспроизведение

```bash
uv sync
uv run python tests/test_trajectory_logic.py
uv run python experiments/run_tier1.py
uv run python experiments/run_tier2.py
uv run python experiments/run_locomo.py
uv run python experiments/run_relational.py
uv run python experiments/train_learned_scorer.py
uv run python experiments/train_learned_scorer_relational.py
uv run python analysis/make_plots.py
uv run python analysis/stats_tests.py
```

Или через Docker:

```
cd docker && docker compose run --rm <service>
```

`run_locomo.py` требует `realdata/data/locomo10.json`:

```bash
mkdir -p realdata/data
curl -sL https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o realdata/data/locomo10.json
```

`run_longmemeval.py` требует `realdata/data/longmemeval_s.json` (не
запускался в этой работе на CPU):

```bash
mkdir -p realdata/data
curl -L --retry 8 --retry-delay 3 --connect-timeout 15 --max-time 900 \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json \
  -o realdata/data/longmemeval_s.json
```

Всё CPU-only и полностью зафиксировано по сидам.

## Дополнительные ограничения и бэклог

**Ограничения, не вошедшие в отчёт:**
- Контроль 1 (§4.5) теряет диагностическую силу при малых `p`:
  при доминировании continuation-шагов почти все кандидаты голосуют за одно
  и то же действие независимо от (перемешанной) метки.
- LoCoMo (§6.5) не имеет собственных негативных контролей -- уверенность в
  корректности пайплайна унаследована от синтетического исследования.
- Обученный combiner (§6.6-6.7) обучен один раз на архитектуру/режим, без
  подбора гиперпараметров и кросс-валидации по сидам. McNemar-тесты
  скорректированы по Holm-Bonferroni внутри каждого эксперимента отдельно,
  но не совместно между экспериментами.
- Пространство действий (6) и алфавит правил малы; поведение на большем
  масштабе не проверялось. Истинное действие зависит только от последнего
  перехода (`k_env=1`), более глубокая история не проверялась.
- Нет сравнения с самим DeMem как бейзлайном -- он решает другую задачу
  (сжатие истории в K сертифицированных слотов с regret-гарантиями), а не
  retrieval из практически неограниченного хранилища.

**Бэклог (топ-3 -- §9):**
- Стратифицированный контроль 1: перемешивать метки корректности отдельно
  по классам действий, а не по всей памяти сразу.
- Экспоненциально взвешенное окно направления вместо ступенчатого `k`
  (H1, §6.2, отклонено для фиксированного `k` -- аналогия с адаптацией
  метилирования в рецепторах бактериального хемотаксиса).
- Шум в сигнале качества `r_t`: инвертировать случайную долю и посмотреть,
  какой метод деградирует быстрее (открытый вопрос из Scope and Limitations
  самой статьи DeMem).
- Больше точек оценки Tier 2, чтобы приблизить его статистическую мощность
  к Tier 1 (§6.4).
- Отдельные негативные контроли для LoCoMo, аналогичные §4.5.
