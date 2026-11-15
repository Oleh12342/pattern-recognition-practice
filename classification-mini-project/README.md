# Порівняння класичних моделей класифікації

## Постановка задачі

За характеристиками пацієнта визначити, чи є ознаки серцевого захворювання. Це задача бінарної класифікації. Позитивний клас: 1 (ознаки наявні). Основна метрика: F1 для позитивного класу; додаткові: Recall, ROC AUC.

## Набір даних

Heart Disease (UCI), підмножина Cleveland (`processed.cleveland.data`): 303 об'єкти, 13 ознак, цільова змінна `target`.
- Числові ознаки: age, trestbps, chol, thalach, oldpeak.
- Категоріальні ознаки: sex, cp, fbs, restecg, exang, slope, ca, thal.
- Цільова змінна: `num` бінаризовано (0 → 0, 1–4 → 1). Класи: 164 / 139. Пропусків 6 (ca: 4, thal: 2).

Готового `heart_disease.csv` не було, тому файл створено з оригінального архіву UCI.

## Методика експерименту

- фіксований стратифікований train/test split 80/20 (`random_state=42`);
- підбір моделей і параметрів лише на train;
- єдина `StratifiedKFold(5, shuffle=True, random_state=42)` для всіх моделей;
- одноразове фінальне оцінювання на test.

## Підготовка даних

Усе всередині Pipeline: `SimpleImputer` (медіана / мода), `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`, `ColumnTransformer`.

## Досліджені моделі

Logistic Regression (L1/L2), k-NN, Gaussian Naive Bayes, LDA, QDA, SVM (linear і RBF).

## Порівняння за крос-валідацією

| Модель | Параметри | CV F1 | CV ROC AUC |
|---|---|---:|---:|
| Logistic Regression | L2, C=10 | 0,8363 | 0,9002 |
| k-NN | k=3, manhattan | 0,8175 | 0,8768 |
| GaussianNB | – | 0,7495 | 0,8773 |
| LDA | solver=svd | 0,8128 | 0,8993 |
| QDA | reg_param=0.1 | 0,7962 | 0,8680 |
| SVM | RBF, C=10, gamma=0.01 | 0,8270 | 0,8968 |

## Фінальна модель і test-результат

Фінальна модель: логістична регресія (L2, C=10). Обрана за найвищим CV F1, ROC AUC і Recall та найменшим std між фолдами (0,0127); проста й інтерпретована.

Test (одноразово): Accuracy = 0,869, Precision = 0,833, Recall = 0,893, F1 = 0,862, ROC AUC = 0,962. Test узгоджується з CV; через малий розмір test (61 об'єкт) оцінка шумна.

## Структура репозиторію

- `data/` – `heart_disease.csv`
- `notebooks/` – `PR04_Kulesh_O.ipynb`
- `models/` – `best_classifier.joblib`
- `reports/` – `cv_model_comparison.csv`, `final_test_metrics.csv`

## Запуск

Відкрити Notebook у Google Colab і послідовно виконати всі комірки.
