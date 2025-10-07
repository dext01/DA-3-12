import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_and_process_data(file_path):
    """
    Загружает данные из CSV, убирает NaN и преобразует в числовой формат.
    :param file_path: Путь к файлу CSV.
    :return: Обработанные данные X и y.
    """
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Убираем целевую переменную и столбец с датой
    X = data.drop(columns=['cnt', 'dteday'])  # Убираем 'cnt' и 'dteday'
    y = data['cnt']  # Целевая переменная - количество арендованных велосипедов

    # Преобразуем X в числовой формат, если есть текстовые данные
    X = X.apply(pd.to_numeric, errors='coerce')

    # Заполнение NaN значений средним
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

    return X, y

def create_windows(data, target, window_size):
    """
    Разбивает данные на окна.
    :param data: Признаки.
    :param target: Целевая переменная.
    :param window_size: Размер окна.
    :return: windows_X, windows_y
    """
    windows_X, windows_y = [], []
    for i in range(len(data) - window_size):
        windows_X.append(data.iloc[i:i+window_size].values)
        windows_y.append(target.iloc[i+window_size])  # Целевое значение после окна
    windows_X = np.array(windows_X)
    windows_y = np.array(windows_y)
    return windows_X, windows_y

def simple_model(window_X):
    """
    Прогнозирует среднее значение всех признаков в окне.
    :param window_X: Окно данных.
    :return: Прогноз (среднее значение).
    """
    return np.mean(window_X)

def calculate_mae(windows_X, windows_y):
    """
    Рассчитывает MAE для каждого окна.
    :param windows_X: Признаки.
    :param windows_y: Целевая переменная.
    :return: Список значений MAE.
    """
    mae_values = []
    for i in range(len(windows_X)):
        actual = windows_y[i]
        predicted = simple_model(windows_X[i])
        mae = mean_absolute_error([actual], [predicted])
        mae_values.append(mae)
    return mae_values

def plot_mae(mae_values):
    """
    Строит график MAE по окнам.
    :param mae_values: Список значений MAE.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mae_values, marker='o', linestyle='-', color='b')
    plt.title("Изменение MAE по окнам", fontsize=14)
    plt.xlabel("Номер окна", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.grid(True)
    plt.show()

def linear_regression_model(windows_X, windows_y):
    """
    Обучает линейную регрессию на данных и оценивает MAE.
    :param windows_X: Признаки.
    :param windows_y: Целевая переменная.
    :return: MAE для линейной регрессии.
    """
    X_train = windows_X.reshape(-1, windows_X.shape[1] * windows_X.shape[2])  # Преобразуем в одномерный массив
    y_train = windows_y.flatten()  # Целевая переменная

    # Обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Прогнозирование и оценка MAE
    predictions = model.predict(X_train)
    mae_linear = mean_absolute_error(y_train, predictions)

    return mae_linear
