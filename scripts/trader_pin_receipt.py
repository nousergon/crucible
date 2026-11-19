"""Acknowledge a completed event-driven pin workflow; cron cannot write this receipt."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol
from urllib.parse import urlsplit

from trader_pin_guard import STATE_MACHINE_ARN_VAR, dispatch_execution


class ReceiptClient(Protocol):
    def put_object(self, *, Bucket: str, Key: str, ContentType: str, Body: bytes) -> object: ...


def main(client: ReceiptClient | None = None) -> int:
    if os.environ.get("GITHUB_EVENT_NAME") != "repository_dispatch":
        raise ValueError("only repository_dispatch can acknowledge a reconcile")
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as handle:
        execution = dispatch_execution(json.load(handle), os.environ[STATE_MACHINE_ARN_VAR])
    uri = urlsplit(os.environ["STORE_URI"])
    if uri.scheme != "s3" or not uri.netloc:
        raise ValueError("receipt requires the configured S3 store")
    digest = hashlib.sha256(execution.encode()).hexdigest()
    key = uri.path.strip("/") + "/runs/release.pin_apply/dispatch/" + digest + ".json"
    if client is None:
        import boto3

        client = boto3.client("s3")
    client.put_object(
        Bucket=uri.netloc,
        Key=key,
        ContentType="application/json",
        Body=json.dumps(
            {
                "schema_version": 1,
                "execution_arn": execution,
                "workflow_run_id": os.environ["GITHUB_RUN_ID"],
                "outcome": "workflow_completed",
            }
        ).encode(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
