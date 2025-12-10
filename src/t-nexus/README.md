Project tree

```
t-nexus/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── auth.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth_router.py
│   │   ├── overview_router.py
│   │   ├── metrics_router.py
│   │   ├── manual_review_router.py
│   │   └── notifications_router.py
│   └── services/
│       ├── __init__.py
│       └── placeholder_data.py
├── frontend/
│   ├── index.html
│   ├── login.html
│   └── register.html
├── ml/
│   ├── # я отказываюсь описывать это на трезвую голову
│   └── # но там hipporag
├── judge/
│   └── batched_generation.py
├── db/
│   └── README.md
├── run_bot.py
└── README.md <-- you are here
```