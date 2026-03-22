# Security policy

This document describes how we handle security reports for **Unmask** and what versions we consider in scope. It complements [CONTRIBUTING.md](CONTRIBUTING.md): contributors should follow both.

---

## Supported versions

This project does not yet publish numbered releases on a fixed schedule. Security fixes land on the default branch (**`main`**) when applicable.

| Ref | Supported with security fixes |
| --- | ----------------------------- |
| Latest `main` | Yes |
| Older commits / forks | Best effort only |

When maintainers tag releases (e.g. `v1.0.0`), this table will be updated to list supported tags and end-of-life rules.

---

## Reporting a vulnerability

**Do not** open a **public** issue or discussion for an undisclosed security vulnerability. That can put users at risk before a fix exists.

### Preferred: private reporting

1. On GitHub, open the repository **Security** tab (if available).
2. Use **Report a vulnerability** / **Private vulnerability reporting** to send details only to maintainers.

If private reporting is not enabled for this repo, maintainers should enable it under **Settings → Security → Code security → Private vulnerability reporting**.

### What to include

- Description of the issue and impact (confidentiality, integrity, availability).
- Steps to reproduce, or a minimal proof of concept.
- Affected component (e.g. Gradio UI, `app.py`, `training/`, dependency).
- Your suggestion for a fix, if you have one.

### What we will do

- **Acknowledge** receipt within **7 days** where possible.
- **Work on a fix** or mitigation; timeline depends on severity and maintainer capacity.
- **Coordinate disclosure** with you before a public advisory or release note, unless you request otherwise or the issue is already public.

We may decline reports that are out of scope (see below) or purely theoretical without a practical impact on this codebase.

---

## Scope (in scope)

Examples of issues we care about:

- Remote code execution in the app or training pipeline when processing untrusted inputs.
- Unsafe deserialization (e.g. loading untrusted `.pth` checkpoints).
- Path traversal or arbitrary file read/write via CLI or UI where the app exposes paths.
- Dependency vulnerabilities with a clear upgrade path in `requirements.txt`.
- Misconfiguration that exposes the Gradio server beyond intended use (documented hardening guidance).

---

## Out of scope

- **Social engineering** or issues with your OS / Python install outside this repo.
- **Denial of service** via extremely large images unless trivially fixable.
- **Theoretical** flaws in upstream libraries without a demonstrated exploit path through Unmask.
- **Privacy** of face images processed locally: users control their data; follow [CONTRIBUTING.md](CONTRIBUTING.md) ethics. (Serious privacy bugs in *our* code paths are still in scope.)

---

## Security expectations for contributors

These practices help maintainers review contributions safely:

- **No secrets** in commits (API keys, tokens, personal paths). Use environment variables and local gitignored config.
- **No large or sensitive** face datasets in PRs; keep data local per `.gitignore`.
- **Pin or review** new dependencies; prefer minimal, maintained packages.
- **Disclose** if you are unsure whether a change has security implications so reviewers can assess it.

Maintainers may request changes or reject PRs that expand attack surface without tests or documentation.

---

## Contact

If GitHub private reporting is unavailable, maintainers may list a dedicated security email in this file. Until then, use GitHub Security Advisories only, or open a **non-sensitive** discussion about enabling a contact address.

---

*This policy may evolve. The version on `main` is authoritative.*
