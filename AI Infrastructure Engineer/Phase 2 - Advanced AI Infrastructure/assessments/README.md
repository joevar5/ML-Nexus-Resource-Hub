# Assessments

This directory holds **track-level** assessments that span multiple modules. Per-module quizzes live alongside their module under `lessons/mod-NN-*/quizzes/`.

## Layout

```
assessments/
├── quizzes/              Track-level quizzes; per-module quizzes are in lessons/
│   └── answer-keys/      Answer key index
├── coding-challenges/    Multi-module coding challenges
├── practical-exams/      Midterm and final practical exams
└── rubrics/              Grading rubrics for projects and exams
```

## Per-module quizzes (canonical locations)

Each module owns its own quiz file. The canonical location is `lessons/mod-NN-*/quizzes/module-quiz.md`. Module 02 additionally has a `mid-module-quiz.md` and `final-quiz.md` covering different lesson ranges.

| Module | Path |
|---|---|
| 02 | `lessons/mod-02-cloud-computing/quizzes/` (mid, final, module) |
| 03 | `lessons/mod-03-containerization/quizzes/module-quiz.md` |
| 04 | `lessons/mod-04-kubernetes/quizzes/module-quiz.md` |
| 05 | `lessons/mod-05-data-pipelines/quizzes/module-quiz.md` |
| 06 | `lessons/mod-06-mlops/quizzes/module-quiz.md` |
| 07 | `lessons/mod-07-gpu-computing/quizzes/module-quiz.md` |
| 08 | `lessons/mod-08-monitoring-observability/quizzes/module-quiz.md` |
| 09 | `lessons/mod-09-infrastructure-as-code/quizzes/module-quiz.md` |
| 10 | `lessons/mod-10-llm-infrastructure/quizzes/module-quiz.md` |

## Track-level assessments here

| File | Purpose |
|---|---|
| `coding-challenges/challenge-01-model-api.md` | Multi-module coding challenge: build a model API spanning Docker + Kubernetes + monitoring. |
| `practical-exams/midterm-practical-exam.md` | Mid-track practical exam covering modules 02–05. |
| `rubrics/project-assessment-rubric.md` | Rubric used to grade all three end-of-track projects. |

## Grading

Each quiz and exam in this repo includes its own pass/fail threshold (typically 80%). For project rubrics, see `rubrics/project-assessment-rubric.md`.
