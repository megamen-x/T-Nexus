"""
RabbitMQ helpers for emitting document lifecycle events.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

from src.t_nexus.ml.config.schema import RabbitMQSettings

try:
    import pika
except ImportError:  # pragma: no cover - optional dependency
    pika = None  # type: ignore[assignment]


class RabbitMQPublisher:
    """Fire-and-forget publisher that emits JSON events."""

    def __init__(self, settings: RabbitMQSettings) -> None:
        """Initialize the RabbitMQ publisher."""
        self.settings = settings
        self._connection = None
        self._channel = None
        if not settings.enabled:
            return
        if pika is None:  # pragma: no cover - import guard
            raise RuntimeError("pika is not installed but RabbitMQ is enabled.")
        params = pika.URLParameters(settings.url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.exchange_declare(
            exchange=settings.exchange,
            exchange_type="topic",
            durable=True,
        )

    def publish(self, routing_key: str, payload: Dict) -> None:
        """Publish *payload* with the specified routing key."""
        if not self.settings.enabled or self._channel is None:
            return
        body = json.dumps(payload).encode("utf-8")
        self._channel.basic_publish(
            exchange=self.settings.exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(content_type="application/json"),
        )

    def document_indexed(self, document_id: str, metadata: Optional[Dict] = None) -> None:
        """Publish a standardized 'document indexed' event."""
        payload = {"document_id": document_id, "metadata": metadata or {}}
        self.publish(self.settings.routing_key_indexed, payload)

    def document_deleted(self, document_id: str) -> None:
        """Publish a standardized 'document deleted' event."""
        payload = {"document_id": document_id}
        self.publish(self.settings.routing_key_deleted, payload)

    def close(self) -> None:
        """Close the RabbitMQ connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._channel = None


class RabbitMQWorker:
    """
    Utility consumer that executes callbacks for incoming messages.
    """

    def __init__(self, settings: RabbitMQSettings, callback: Callable[[Dict], None]) -> None:
        """Create a blocking consumer that dispatches payloads to *callback*."""
        if pika is None:  # pragma: no cover - import guard
            raise RuntimeError("pika is not installed but RabbitMQ worker is requested.")
        self.settings = settings
        self.callback = callback
        params = pika.URLParameters(settings.url)
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=settings.queue, durable=True)
        self._channel.queue_bind(
            exchange=settings.exchange,
            queue=settings.queue,
            routing_key="#",
        )

    def _on_message(self, ch, method, properties, body) -> None:
        """Deserialize payload and invoke the user callback."""
        payload = json.loads(body.decode("utf-8"))
        self.callback(payload)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self) -> None:
        """Start consuming messages."""
        self._channel.basic_qos(prefetch_count=1)
        self._channel.basic_consume(queue=self.settings.queue, on_message_callback=self._on_message)
        self._channel.start_consuming()

    def stop(self) -> None:
        """Stop consuming and close the connection."""
        self._channel.stop_consuming()
        self._connection.close()
