# Security Policy

## Reporting a vulnerability

Report privately through GitHub Security Advisories on this repository
("Security" → "Report a vulnerability"). Do not open a public issue for a
suspected vulnerability.

Expect an acknowledgement within 5 business days and an assessment within 15.

## Scope

This repository is an experiment harness. It holds no credentials, no
account identifiers and no strategy content: every secret is read at runtime
from the environment or from AWS SSM Parameter Store, and strategy
configuration is loaded from a private source outside this repository.

In scope: the CLI, the run-manifest schema, the store abstraction, the slot
registry and the alerting path. Out of scope: third-party dependencies
(report upstream), and any deployment operated by someone other than the
maintainers.

## Supported versions

The `main` branch only. Releases are immutable, content-addressed artifacts;
a fix ships as a new release and a pointer flip, never as a patch to a
published artifact.
