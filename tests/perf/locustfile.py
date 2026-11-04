"""Load test:  locust -f tests/perf/locustfile.py --host http://localhost:8000 -u 5 -r 1 -t 60s --headless

Set API_KEY in the environment. Compare p50/p95 of the /ask rows with roadmap section 2.1.
"""

import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "How many days of annual leave do employees get?",
    "How long is maternity leave?",
    "What is the hotel limit in London?",
    "What is the meal allowance for international trips?",
    "What is the capital of France?",
]


class AskUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.client.headers.update({"X-API-Key": os.environ.get("API_KEY", "dev-key-change-me")})

    @task(5)
    def ask(self) -> None:
        q = random.choice(QUESTIONS) + " " * random.randint(0, 3) + random.choice(["", "?", " please"])
        self.client.post("/ask", json={"question": q}, name="/ask")

    @task(1)
    def list_documents(self) -> None:
        self.client.get("/documents", name="/documents")
