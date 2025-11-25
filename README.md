<a name="readme-top"></a>  
<img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/main/assets/tnexus_banner.png" alt="megamen banner">
<div align="center">
  <p align="center">
  </p>
  <p align="center">
    <strong>T-Nexus</strong> - 
    RAG по базе данных Т-Бизнес
    <p></p>
    Создано <strong>megamen</strong>, совместно с <br /> <strong> AI Talent Hub</strong>
    <br /><br />
    <a href="https://github.com/megamen-x/T-Nexus/issues" style="color: black;">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/megamen-x/T-Nexus/discussions/1" style="color: black;">Предложить улучшение</a>
  </p>
</div>

**Содержание:**
- [Проблематика задачи](#title1)
- [Описание решения](#title2)
- [Тестирование решения](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика задачи</a></h3> 
В рамках курса "Введение в большие языковые модели" необходимо создать работающего бота-помощника. Функционал: генеративные ответы на текстовые запросы с указанием источников, а также админская дашборд-панель для мониторинга показателей.

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**Общая схема решения:**

<img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/main/assets/tnexus_arch.png" alt="t-nexus architecture">

**Клиентская часть**

[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/)
[![HTML](https://img.shields.io/badge/HTML-%23E34F26.svg?style=for-the-badge&logo=html)](#)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-%2338B2AC.svg?style=for-the-badge&logo=tailwind)](#)

**Серверная часть**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![SQLite](https://img.shields.io/badge/SQLite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)](#)


<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title3">Тестирование решения</a></h3> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:
  
  **```Telegram-bot + Dasshboard + FastAPI + ML-models;```**

<details>
  <summary> <strong><i> Инструкция по запуску FastAPI-сервера:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell activation recommended**) через терминал последовательно выполнить следующие команды:
  
    - Клонирование репозитория:
    ```
    git clone https://github.com/megamen-x/T-Nexus.git
    ```
    - Создание и активация виртуального окружения:
    ```
    cd ./T-Nexus
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей:
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install -r requirements.txt
    ```
    - После установки зависимостей (5-7 минут):
    ```
    cd ./t-nexus/backend
    uvicorn main:app --reload --host localhost --port 8000
    ```
</details> 

</br> 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>
