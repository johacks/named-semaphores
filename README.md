# Pythonic API for POSIX IPC Named Semaphores

![License](https://img.shields.io/github/license/johacks/named-semaphores)
![PyPI Version](https://img.shields.io/pypi/v/named-semaphores)
[![Coverage Status](https://codecov.io/gh/johacks/named-semaphores/branch/main/graph/badge.svg)](https://codecov.io/gh/johacks/named-semaphores)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0c27d0f8d27a468598813292178b4881)](https://app.codacy.com/gh/johacks/named-semaphores/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

## Description

Use POSIX IPC named semaphores with a high-level API similar to Python's `threading.Semaphore`. This repository wraps around existing bindings from the [posix_ipc](https://pypi.org/project/posix-ipc/) package to provide a more Pythonic interface.

## Features

- **Named Semaphores**: Handle POSIX IPC named semaphores with ease.
- **Cross-Platform**: Works on Linux, macOS, and Windows (via Cygwin).
- **Thread-Safe**: Built on POSIX IPC, ensuring robust multi-process handling.
- **Pythonic API**: Similar to Python's built-in `threading.Semaphore` for familiarity.
- **Flexible Creation**: Choose how to handle existing semaphores (`RAISE_IF_EXISTS`, `LINK_OR_CREATE`, etc.).
- **Timeouts**: Optionally specify timeouts for acquiring semaphores (platform-dependent).
- **Automatic Cleanup**: Semaphore can be automatically unlinked when the object is deleted.

## Installation

You can install the package from PyPI:

```bash
pip install named-semaphores
```

## Usage

The `NamedSemaphore` class provides a high-level API for working with POSIX IPC named semaphores. Below are various usage examples demonstrating its flexibility and functionality.

### 1. **Basic Semaphore Creation**

Create a semaphore with a default initial value of 1.

```python
from named_semaphores import NamedSemaphore

sem = NamedSemaphore("example_semaphore")
print(f"Semaphore '{sem.name}' created.")
```

You can also specify a custom initial value for the semaphore.

```python
sem = NamedSemaphore("example_semaphore", initial_value=3)
print(f"Semaphore '{sem.name}' created with initial value of 3.")
```

---

### 2. **Acquire, Release, and Timeouts**

You can acquire and release the semaphore, specify a custom initial value, or use a timeout for acquiring (if supported on your platform).

#### Acquire and Release

```python
sem = NamedSemaphore("example_semaphore")

# Acquire the semaphore
sem.acquire()
try:
    print("Critical section protected by semaphore.")
finally:
    # Release the semaphore
    sem.release()
    print("Semaphore released.")
```

#### Acquire with Timeout

```python
# Acquire the semaphore with a 5-second timeout
acquired = sem.acquire(timeout=5)
if acquired:
    print("Semaphore acquired within timeout.")
    sem.release()
else:
    print("Failed to acquire semaphore within timeout.")
```

---

### 3. **Using as a Context Manager**
The semaphore can be used in a with statement to automatically acquire and release it.

```python
with NamedSemaphore("example_semaphore") as sem:
    print(f"Semaphore '{sem.name}' acquired.")
    # Critical section here
print(f"Semaphore '{sem.name}' released.")
```

---

### 4. **Unlinking a Semaphore**

Unlinking a semaphore removes it from the system after all processes using it have closed it. Note some comments made by the `posix_ipc` author:

```markdown
“If any processes have the semaphore open when unlink is called, the call to unlink returns immediately but destruction of the semaphore is postponed until all processes have closed the semaphore.

Note, however, that once a semaphore has been unlinked, calls to open() with the same name should refer to a new semaphore. Sound confusing? It is, and you'd probably be wise structure your code so as to avoid this situation.”
```
By default, semaphores created by this class are automatically unlinked when the object is deleted. You can override this behavior with the `unlink_on_delete` parameter.

- Automatic unlinking example:

```python
# Create a semaphore
handle_create = NamedSemaphore(
    "example_semaphore",
    handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS
)
# Link to the semaphore
handle_link = NamedSemaphore(
    "example_semaphore",
    handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
)
# Trigger garbage collection
del handle_link  # Only closes the handle, does not unlink the semaphore
del handle_create  # Unlinks the semaphore, because this handle created it
```
- Manual unlinking example:

```python
# Set `unlink_on_delete` to False to prevent automatic unlinking on object deletion
sem = NamedSemaphore("example_semaphore", unlink_on_delete=False)
del sem  # Semaphore is not unlinked
sem = NamedSemaphore("example_semaphore", unlink_on_delete=False)
# Unlink the semaphore manually, handle will be closed in the destructor
sem.unlink()
```

---

### 5. **Handling of Existing Semaphores**

The `NamedSemaphore` class provides flags to handle existing semaphores in different ways. The safest ways are the `RAISE_IF_EXISTS` and `RAISE_IF_NOT_EXISTS` flags, which raise an error if the semaphore already exists or does not exist, respectively.

Some other flags are `LINK_OR_CREATE` and `UNLINK_AND_CREATE`, which link to an existing semaphore or unlink it if it already exists. They may be useful in certain scenarios, but should be used with caution, as they may silently hide issues in the code (e.g., not properly cleaning up semaphores).

#### Raise an error if the semaphore already exists

Use the `RAISE_IF_EXISTS` flag to ensure a new semaphore is created and raise an error if one already exists.

```python
try:
    sem = NamedSemaphore(
        "example_semaphore",
        handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS
    )
    print(f"Semaphore '{sem.name}' created without existing conflict.")
except FileExistsError:
    print("Handling existing semaphore error...")
```

#### Link to a semaphore or create it if it doesn't exist

Use the `LINK_OR_CREATE` flag to link to an existing semaphore or create a new one if it doesn't exist.

```python
sem = NamedSemaphore(
    "example_semaphore",
    handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE
)
print(f"Semaphore '{sem.name}' linked or created.")
```

#### Link to a semaphore or raise an error if it doesn't exist

Use the `RAISE_IF_NOT_EXISTS` flag to link to an existing semaphore, raising an error if it does not exist.

```python
try:
    sem = NamedSemaphore(
        "example_semaphore",
        handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
    )
    print(f"Semaphore '{sem.name}' linked.")
except FileNotFoundError:
    print("Handling non-existing semaphore error...")
```

#### Create a Semaphore, unlinking it if it already exists

Use the `UNLINK_AND_CREATE` flag to unlink semaphore if it already exists and create a new one.

```python
sem = NamedSemaphore(
    "example_semaphore",
    handle_existence=NamedSemaphore.Flags.UNLINK_AND_CREATE
)
print(f"Semaphore '{sem.name}' deleted and created.")
```

**Note**: although convenient, rather than using a forced deletion and create, it is better practice to handle the existence of semaphores in a more controlled manner. An already existing semaphore usually indicates that it is being used by another process, or that it was not properly cleaned up by a previous process.

---

These examples cover the most common scenarios for using NamedSemaphore. For advanced use cases or troubleshooting, refer to the module's documentation or docstrings.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

Please note that this project depends on the [posix_ipc](https://pypi.org/project/posix-ipc/) package, which is licensed under a BSD-style license. For details about the `posix_ipc` license, refer to its [license file](https://raw.githubusercontent.com/osvenskan/posix_ipc/4db678001be2f16175c70cb88d4fb9f9126333f5/LICENSE).
