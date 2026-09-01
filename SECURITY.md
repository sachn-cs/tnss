# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within TNSS, please send an email to **sachn.cs@gmail.com**. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to include

When reporting a vulnerability, please include:

- Type of vulnerability (e.g., buffer overflow, code injection, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will investigate and validate the reported vulnerability.
- **Resolution**: We will work on a fix and coordinate disclosure timing with you.
- **Disclosure**: Once a fix is available, we will publicly disclose the vulnerability.

## Security Best Practices

When using TNSS in your project:

- Keep your Rust toolchain updated to the latest stable version
- Run `cargo audit` regularly to check for known vulnerabilities in dependencies
- Review `deny.toml` for the project's dependency policy
- Use `cargo deny` to enforce license and vulnerability checks

## Scope

This security policy applies to:

- The TNSS Rust crates
- The command-line interface (`tensift-cli`)
- Documentation and examples in this repository

## Out of Scope

- Third-party dependencies (report these to their respective maintainers)
- Issues in development/unreleased versions
- Issues requiring physical access to the machine

## Recognition

We appreciate the efforts of security researchers and will acknowledge contributors who help improve the security of this project (unless they prefer to remain anonymous).
