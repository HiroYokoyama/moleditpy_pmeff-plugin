# Contributing Guidelines

Thank you for your interest in contributing! We welcome contributions from the community to make this project more robust and feature-rich.

To maintain the high quality and stability of the application, please review the following guidelines.

## 1. How to Report Bugs

When opening an Issue, please include:
1.  **Steps to Reproduce:** A detailed list of actions taken.
2.  **Expected Behavior vs. Actual Behavior.**
3.  **Logs / Traceback:** If applicable, please attach the contents of the console output or log files.
4.  **Environment:** OS, Python version, library/package version, etc.

## 2. Development Setup

1.  Clone the repository.
2.  Copy or symlink the plugin package folder into your **user plugin directory** of the main **MoleditPy** application so it is discovered automatically at startup:
    - **Windows**: `C:\Users\<YourName>\.moleditpy\plugins\`
    - **Linux / macOS**: `~/.moleditpy/plugins/`
3.  Install development dependencies as required by the project.

## 3. Coding Standards

We maintain high code quality standards. Before submitting a Pull Request (PR), ensure your code meets the following:

* **Style:** Follow standard style guidelines (e.g. PEP 8 for Python).
* **Type Hinting:** Use type hints for function arguments and return values where applicable.

## 4. Testing & Stability Policy

### A. Core Logic & Testing
* Core logic and utility modules should have automated unit tests.
* Maintain high test coverage for any new features.

### B. Error Handling — No Hiding, No Crashing
* Stability policy: **never hide errors, never crash.**
* Wrap UI slots, event handlers, or callbacks in `try/except` block, log the traceback, and show a status message or dialog to the user.
* Internal helpers should propagate exceptions rather than swallowing them.
* Write logs/tracebacks on caught exceptions; empty `except` blocks are not acceptable.

### C. GUI & Interaction
* **Avoid Broad Exceptions:** Minimize the use of broad `try-except Exception` blocks in internal helper code. Prefer granular, specific exception types.
* **Manual Verification:** Verify GUI changes manually and document your verification in the PR.

## 5. Pull Request Process

1.  **Branching:** Create a new branch for your feature or fix.
2.  **Verification:**
    * Run tests and check linting before submitting.
3.  **Description:** Describe your changes clearly. If it's a UI change, screenshots are highly appreciated.
4.  **Review:** Wait for a maintainer to review your code.

## 6. License

By contributing, you agree that your contributions will be licensed under the project's license.
