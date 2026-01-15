<a name="readme-top"></a>  
<!-- <img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/main/assets/t_nexus_banner.png" alt="megamen banner"> -->
<img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/html_testing/assets/t_nexus_banner.png" alt="megamen banner">
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
- [Проблематика задачи](#проблематика-задачи)
- [Описание решения](#описание-решения)
- [Тестирование решения](#тестирование-решения)
- [Обновления](#обновления)
- [Команда](#команда)

## <h3 align="start"><a id="title1">Проблематика задачи</a></h3> 
В рамках курса "Введение в большие языковые модели" необходимо создать работающего бота-помощника. Функционал: генеративные ответы на текстовые запросы с указанием источников, а также админская дашборд-панель для мониторинга показателей.

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Пример работы:**

https://github.com/user-attachments/assets/32361304-f046-44ef-8464-a1af3d432e90

**Обработка данных:**

Поля датасета:
* **title**: Заголовок статьи или формулировка вопроса. Используется для быстрого поиска и ранжирования.
* **url**: Ссылка на источник. Необходима для цитирования в ответе LLM и проверки актуальности.
* **description**: Основное тело статьи или ответ на вопрос.

Ссылка на сэмпл собранных данных (200 строк от всего объема): 
https://drive.google.com/file/d/1FX3fw95I25ZXZTsWsgpcxJiVebxllx5g/view?usp=sharing 

Распределение данных:

<!-- <img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/main/assets/t_nexus_data.png" alt="t-nexus dataset"> -->
<img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/html_testing/assets/t_nexus_data.png" alt="t-nexus dataset">


**Архитектура сервиса:**

<!-- <img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/main/assets/t_nexus_arch.png" alt="t-nexus architecture"> -->
<img width="100%" src="https://github.com/megamen-x/T-Nexus/blob/html_testing/assets/t_nexus_arch.png" alt="t-nexus architecture">

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

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
  
  **```Telegram-bot + Dashboard + FastAPI + ML-models;```**

<details>
  <summary> <strong><i>Локальный запуск решения:</i></strong> </summary>
  
  - В Visual Studio Code через терминал последовательно выполните следующие команды:

  - 1. Склонируйте репозиторий:

  ```
  git clone https://github.com/megamen-x/T-Nexus.git
  ```

  - 2. Создайте окружение и установите зависимости проекта:

  ```
  uv venv .venv
  source .venv/bin/activate
  uv pip install -e .
  ```

  - 3. Скопируйте содержимое файла .env.example в файл .env 
  - 4. Настройте доступ к хосту LLM-модели:
  
  ```
  OPENAI_BASE_URL=https://host.llm/api
  OPENAI_API_KEY=openai_api_key
  ```

  - 5. Добавьте ключ Telegram-бота:
  
  ```
  BOT_TOKEN=telegram_bot_token
  ```

  - 6. После окончания предыдущих этапов можно запустить сервер:
  
  ```
  uvicorn src.t_nexus.main:app --reload --host localhost --port 8000
  ```

</details> 

<details>

<summary> <strong><i>Запуск решения при помощи docker compose:</i></strong> </summary>
  
  - Примечание:

  Для корректной работы T-Nexus необходим Python 3.12, предпочтительная версия CUDA - 12.9. 

  - В Visual Studio Code через терминал последовательно выполните следующие команды:

  - 1. Склонируйте репозиторий:

  ```
  git clone https://github.com/megamen-x/T-Nexus.git
  ```
  - 2. Скопируйте содержимое файла .env.example в файл .env 
  - 3. Настройте доступ к хосту LLM-модели:
  
  ```
  OPENAI_BASE_URL=https://host.llm/api
  OPENAI_API_KEY=openai_api_key
  ```

  - 4. Добавьте ключ Telegram-бота:
  
  ```
  BOT_TOKEN=telegram_bot_token
  ```

  - 5. После окончания предыдущих этапов запустите сервер:
  
  ```
  docker compose up
  ```

</details>

</br> 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


## <h3 align="start"><a id="title4">Обновления</a></h3> 

***ToDo list***
New feature | WIP | Done |
--- |:---:|:---:|
baseline + tg-bot | &#x2611; | &#x2611; | 
baseline + tg-bot + dashboard | &#x2611; | &#x2610; | 
LLM as a Judge | &#x2611; | &#x2610; | 
Metrics | &#x2610; | &#x2610; | 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title4">Метрики</a></h3> 

<img src="/assets/metrics.png" width="100%">

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title5">Команда</a></h3> 

- [Луняков Алексей](https://github.com/AlexeyLunyakov) - UX\UI, Full-Stack Engineer
- [Калинин Александр](https://github.com/Agar1us) - DL / Full-Stack Engineer
- [Полетаев Владислав](https://github.com/whatisslove11) - ML / DL Engineer
- [Чуфистов Георгий](https://github.com/georgechufff) - ML / DL Engineer, ML Ops

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>
