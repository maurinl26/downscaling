# Security Policy

`karpos-downscaling` is a research-grade scientific Python package for
atmospheric reanalysis downscaling. It is not a security-critical system: it
does not provide authentication, does not ingest data from untrusted users at
runtime, and does not expose network services by default.

However, like any software, it depends on third-party libraries and processes
external scientific data. We take security reports seriously and appreciate
responsible disclosure.

## Supported versions

We follow [Semantic Versioning](https://semver.org/). Security fixes are
applied to the latest minor release of each supported major version.

| Version | Supported |
|---|---|
| 0.x (current development) | ✅ |
| Pre-0.x experimental tags | ❌ |

Once a `1.0` release exists, this table will be updated to reflect long-term
support windows.

## Reporting a vulnerability

If you discover a security vulnerability — for example, an unsafe pickle
deserialization, an arbitrary code execution via crafted input file, a
dependency supply-chain compromise, or a credential leak in published
artefacts — **please do not open a public GitHub issue**.

Instead, please report privately to:

**loic.maurin@karpos.pro**

(use PGP if you wish: a public key will be linked here once published)

In your report, please include:

1. A description of the vulnerability and its potential impact
2. Steps to reproduce, including a minimal example if possible
3. Affected version(s) of `karpos-downscaling`
4. Suggested fix or mitigation if you have one
5. Whether you wish to be credited in the eventual security advisory

## Response process

Our commitment:

- **Acknowledgement** within **5 business days** of receipt
- **Initial assessment** (confirmed / not reproducible / out of scope) within
  **15 business days**
- **Fix and disclosure** coordinated with the reporter, typically within
  **60-90 days** depending on severity

For severe issues affecting users in production, we will work to ship a fix
faster and may issue an interim advisory.

## Disclosure policy

We follow **coordinated disclosure**. We will:

1. Confirm the vulnerability and develop a fix in a private branch
2. Coordinate a disclosure date with the reporter
3. Publish a [GitHub Security Advisory](https://github.com/maurinl26/karpos-downscaling/security/advisories)
   with the fix released as a patch version
4. Credit the reporter in the advisory (if they consent)

If you do not receive a response within the timeframes above, please follow up
on the same email address. If you still receive no response, you may publicly
disclose the issue.

## Scope

In scope:

- Code in this repository (`downscaling/`, `tests/`, `scripts/`, `configs/`)
- Default behavior of published packages
- Documented usage patterns
- Dependencies pinned in `pyproject.toml` and `uv.lock`

Out of scope:

- Vulnerabilities in upstream dependencies (please report those upstream and
  notify us if a coordinated update is needed)
- Issues that require pre-existing local access to a user's environment
- Misconfiguration by end users that violates documented usage
- Vulnerabilities in unrelated repositories under `maurinl26` (please report
  those to the appropriate repository)

## Acknowledgements

We thank all security researchers who help keep open-source scientific
computing safe. Reporters who responsibly disclose vulnerabilities and consent
to being named will be credited in the corresponding security advisory.
