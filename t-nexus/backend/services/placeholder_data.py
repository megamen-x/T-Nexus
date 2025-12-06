from typing import Dict, Any, List

def get_overview_placeholder() -> Dict[str, Any]:
    return {
        "period": "Last 7 Days",
        "kpis": [
            {"label": "Active Dialogs", "value": "3,240", "delta": "+12.4%"},
            {"label": "LLM Coverage", "value": "96.2%", "delta": "+2.1%"},
            {"label": "Avg. Handle Time", "value": "3m 40s", "delta": "-0.5m"},
            {"label": "Escalations", "value": "42", "delta": "-9%"}
        ],
        "conversations": {"total": 1820, "avgDuration": "3m 40s"},
        "successRate": 0.918,
        "traffic": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "values": [220, 280, 260, 310, 420, 280, 360]
        },
        "ratings": {"positive": 64, "neutral": 23, "negative": 13},
        "incidents": [
            {"id": "INC-204", "impact": "Delayed answer on premium tier", "status": "In progress", "timestamp": "08:41"},
            {"id": "INC-205", "impact": "Webhook timeout RabbitMQ cluster B", "status": "Investigating", "timestamp": "09:15"},
            {"id": "INC-198", "impact": "Aiogram bot auth drop for 3 users", "status": "Resolved", "timestamp": "Yesterday"}
        ]
    }

def get_metrics_placeholder() -> Dict[str, Any]:
    return {
        "llm": {
            "requests": 4820,
            "avgTime": 1.84,
            "avgTokens": 812,
            "avgResponseLength": 218,
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
            "positive": [58, 61, 64, 63, 66, 68, 71],
            "negative": [7, 6, 6, 5, 5, 4, 3]
        },
        "hipporag": {
            "relevance": 0.872,
            "topQueries": ["Pricing tiers", "Integration setup", "Reset API key", "Telegram bot limits", "Bulk CSV answers"],
            "slowQueries": [
                {"label": "Historical invoices", "duration": "4.1s"},
                {"label": "Legacy contract ids", "duration": "3.8s"},
                {"label": "Warehouse SLA", "duration": "3.2s"},
                {"label": "CSV sync errors", "duration": "3.1s"},
                {"label": "Partner seat count", "duration": "2.9s"}
            ]
        },
        "user": {
            "topQueries": ["Reset password", "Connect CRM", "Share analytics", "Voice request limits", "CSV templates"],
            "completionRate": 0.884,
            "tagCloud": [
                {"label": "billing", "weight": 5},
                {"label": "voice", "weight": 3},
                {"label": "hipporag", "weight": 2},
                {"label": "telegram", "weight": 4},
                {"label": "csv", "weight": 3},
                {"label": "sources", "weight": 2},
                {"label": "llm", "weight": 4},
                {"label": "support", "weight": 3}
            ]
        },
        "popularQueries": ["How to upload CSV?", "Can I export analytics?", "Where to see sources?", "Voice limits?", "Reset token?"]
    }

def get_manual_review_placeholder() -> List[Dict[str, Any]]:
    return [
        {
            "id": 1044,
            "title": "Bulk pricing for healthcare",
            "question": "Could you confirm whether healthcare enterprise bulk plans include unlimited CSV answers and Telegram voice responses? Need to sync with compliance.",
            "modelResponse": "Yes, healthcare enterprise tier includes unlimited CSV answers, voice transcription within 120 seconds, and top 5 cited sources with HippoRAG context when available.",
            "adminResponse": ""
        },
        {
            "id": 1045,
            "title": "Webhook retry policy",
            "question": "Our RabbitMQ consumer missed messages last night. What is the retry interval configured for T-Nexus outbound events?",
            "modelResponse": "T-Nexus retries webhook delivery three times with 30s exponential backoff and surfaces failed deliveries into Manual Review.",
            "adminResponse": ""
        },
        {
            "id": 1046,
            "title": "Voice locale expansion",
            "question": "When will the Telegram bot support Japanese voice prompts with aiogram? Need ETA for next quarter.",
            "modelResponse": "Voice locale expansion is rolling out in Q4 with Japanese, Korean, and German coverage using the same aiogram orchestrator.",
            "adminResponse": ""
        }
    ]

def get_notifications_placeholder() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "title": "LLM latency baseline", "message": "Average generation time stabilized at 1.8s in the last window.", "ts": "2m ago"},
        {"id": 2, "title": "HippoRAG freshness", "message": "New finance knowledge pack ingested.", "ts": "15m ago"},
        {"id": 3, "title": "Telegram CSV burst", "message": "34 CSV batches served in the last hour.", "ts": "42m ago"}
    ]