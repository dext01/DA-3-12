# DA-3-12 Анализ: Кросс-валидация временных рядов

## Описание задачи

Задача направлена на анализ **стабильности модели** при использовании **кросс-валидации временных рядов**. Основные шаги заключаются в:

1. Разделении временного ряда на 5 последовательных окон.
2. Обучении простой модели (среднее значение) для каждого окна.
3. Оценке модели с использованием метрики **MAE (средняя абсолютная ошибка)**.
4. Построении графика **MAE** для анализа стабильности модели на разных отрезках данных.

Цель задачи — проверить, как изменяется ошибка модели по мере увеличения окон и делать выводы о стабильности модели. Мы ожидаем, что **MAE не будет расти со временем**, если модель стабильна.

## Как запустить код в Google Colab

### Шаг 1: Загрузка репозитория

1. Клонируйте или скачайте этот репозиторий на свой компьютер.
2. Перейдите в [Google Colab](https://colab.research.google.com/) и откройте новый ноутбук.
3. В меню **`File -> Open notebook`** выберите **`GitHub`** и вставьте ссылку на ваш репозиторий или загрузите локальные файлы.

### Шаг 2: Загрузка данных

1. Файл **`bike_sharing_dataset.zip`** можно скачать из GitHub с помощью команды **`curl`** в **Google Colab**:

    ```python
    !curl -O https://raw.githubusercontent.com/dext01/DA-3-12/main/bike_sharing_dataset.zip
    ```

2. Разархивировать файл:

    ```python
    import zipfile

    # Разархивируем файл
    with zipfile.ZipFile('bike_sharing_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('bike_sharing_data')

    # Проверим содержимое распакованной папки
    import os
    print(os.listdir('bike_sharing_data'))
    ```

3. Загрузить данные с помощью **pandas**:

    ```python
    import pandas as pd

    # Загрузка данных из файла 'hour.csv'
    data = pd.read_csv('bike_sharing_data/hour.csv')

    # Проверим первые строки данных
    print(data.head())
    ```

### Шаг 3: Импорт Python функций

1. Скачайте Python файл с GitHub:

    ```python
    !curl -O https://raw.githubusercontent.com/dext01/DA-3-12/main/DA_3_12.py
    ```

2. Импортируйте необходимые функции из этого файла:

    ```python
    from DA_3_12 import load_and_process_data, create_windows, simple_model, calculate_mae, plot_mae, linear_regression_model
    ```

### Шаг 4: Загрузка и обработка данных

1. Укажите правильный путь к файлу данных (например, **`hour.csv`**):

    ```python
    file_path = 'bike_sharing_data/hour.csv'  # Путь к файлу с данными
    X, y = load_and_process_data(file_path)

    # Проверка первых строк
    print(X.head())
    print(y.head())
    ```

### Шаг 5: Разбиение данных на окна

Используйте функцию `create_windows`, чтобы разбить данные на окна:

```python
# Разбиение данных на окна
window_size = 3  # Размер окна
windows_X, windows_y = create_windows(X, y, window_size)

# Проверка размеров окон
print(f"Количество окон: {len(windows_X)}")
```

### Шаг 6: Оценка модели с использованием простой модели (среднее)

Используйте функцию **`calculate_mae`**, чтобы оценить **MAE** для каждого окна:

```python
# Оценка MAE для модели на основе среднего
mae_values = calculate_mae(windows_X, windows_y)

# Выводим MAE
print(mae_values)
```

### Шаг 7: Построение графика MAE
```
# Построение графика MAE
plot_mae(mae_values)
```
![Мой результат для датасета](https://raw.githubusercontent.com/dext01/DA-3-12/main/images/MAE_image.png)

### Шаг 8: Использование линейной регрессии
```
# Оценка MAE с помощью линейной регрессии
mae_linear = linear_regression_model(windows_X, windows_y)

# Выводим MAE для линейной регрессии
print(f"MAE для линейной регрессии: {mae_linear}")
```
