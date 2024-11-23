"""
semaphore.py

This module contains a class to handle a POSIX IPC named semaphore.

:author: 2024 Joaquin Jimenez
"""

from enum import Enum
from numbers import Real
import sys
from typing import Optional

import posix_ipc
from typing_extensions import Self

from easy_posix_ipc.logging import LoggingMixin
import signal


class NamedSemaphore(LoggingMixin):
    """
    Class to handle a POSIX-IPC named semaphore.

    This class provides a Pythonic interface to POSIX named semaphores. It supports multi-process
    environments and relies on the underlying thread-safe POSIX IPC implementation. After creation,
    the semaphore handle is primarily read-only, ensuring thread safety for typical usage.

    To create a new semaphore ensuring that it did not exist before, you can set the `handle_existence`
    parameter to `RAISE_IF_EXISTS`. This will raise an error if the semaphore already exists:

    ```
    # Raises an error if the semaphore already exists
    my_sem = NamedSemaphore("max_api_calls", handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)
    ```

    To create or link a semaphore ignoring the existence of a previous semaphore with the same name,
    you can use the `LINK_OR_CREATE` parameter:

    ```
    my_sem = NamedSemaphore("max_api_calls", handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    ```

    To create a new semaphore deleting the previous one if it exists, you can set the `handle_existence`
    parameter to `DELETE_AND_CREATE`. This will delete the existing semaphore and create
    a new one:

    ```
    my_sem = NamedSemaphore("max_api_calls", handle_existence=NamedSemaphore.Flags.DELETE_AND_CREATE)
    ```

    The class provides a context manager interface, which acquires the semaphore on entry and
    releases it on exit. This is the recommended way to use the semaphore if it is assumed that the
    semaphore was already created. In this case, the RAISE_IF_NOT_EXISTS flag can be used to raise
    an error if the semaphore does not previously exist, ensuring that the semaphore is created
    beforehand:

    ```
    # Raises an error if the semaphore does not yet exist
    with NamedSemaphore("max_api_calls", handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS):
        # Critical section
        ...
    ```

    Unlinking of the semaphore:
    - By default, the semaphore is unlinked by the garbage collector when the object is deleted if
      it was created by this handle. Else, only the descriptor is closed. This behavior can be
      overridden by setting the `unlink_on_delete` parameter in the constructor.
    - The semaphore can also be unlinked manually by calling the `unlink` method. This removes the
      semaphore globally, making it inaccessible by its name.
    - The semaphore can also be unlinked by a signal handler for SIGINT, SIGTERM, and SIGHUP. This
      handler is set up by default if this object creates the semaphore, but can be overridden by
      setting the `unlink_on_signal` parameter in the constructor.
    """

    class Flags(Enum):
        """
        Enum for the flags to handle existing semaphores.
        """

        RAISE_IF_EXISTS = 0
        LINK_OR_CREATE = 1
        RAISE_IF_NOT_EXISTS = 2
        DELETE_AND_CREATE = 3

    def __init__(
        self,
        name: str,
        initial_value: int = 1,
        handle_existence: Flags = Flags.RAISE_IF_NOT_EXISTS,
        unlink_on_delete: Optional[bool] = None,
        unlink_on_signal: Optional[bool] = None,
    ) -> None:
        """
        Create a POSIX IPC named semaphore.

        The `handle_existence` parameter controls the behavior regarding the existence of the semaphore:
        - `RAISE_IF_EXISTS`: Creates a new semaphore, raises an error if it already exists.
        - `LINK_OR_CREATE`: Links to the existing semaphore if it exists.
        - `RAISE_IF_NOT_EXISTS`: Links to the existing semaphore if it exists, raises an error otherwise.
        - `DELETE_AND_CREATE`: Deletes the existing semaphore and creates a new one.

        The semaphore is automatically unlinked when the object is deleted if it was
        created by this handle. Else, the semaphore is only closed.

        :param str name: The name of the semaphore.
        :param int initial_value: The initial value of the semaphore. Default is 1.
        :param NamedSemaphore.Flags handle_existence: Behavior regarding existence of the semaphore.
        :param Optional[bool] unlink_on_delete: If True, the semaphore will be unlinked when the
            object is deleted or garbage collected. If False, the semaphore will only be closed. The
            default is None, which evaluates to True if the semaphore was created by this handle.
        :param Optional[bool] unlink_on_signal: If True, the semaphore will be unlinked when the
            process receives a SIGINT, SIGTERM, or SIGHUP signal. If False, the signal handler will
            not be set up. The default is None, which evaluates the same as `unlink_on_delete`.

        :raises ValueError: If the input parameters are invalid.
        :raises PermissionError: If the semaphore cannot be created due to permissions.
        :raises FileExistsError: If the semaphore already exists and could not be removed after
            setting `handle_existence` to `RAISE_IF_EXISTS`.
        :raises FileNotFoundError: If the semaphore could not be found after setting
            `handle_existence` to `RAISE_IF_NOT_EXISTS`.
        """
        # Clean the name and initialize the logger
        name = name.removeprefix("/") if isinstance(name, str) else ""
        LoggingMixin.__init__(self, name)

        # Check the input parameters
        if not name or not all(c.isalnum() or c in ("-", "_") for c in name):
            raise ValueError(
                "`name` must be a non-empty string with characters '-', '_' or alphanumeric. "
                f"Got: {name}"
            )
        if not (isinstance(initial_value, int) and initial_value >= 0):
            raise ValueError("`initial_value` must be a non-negative integer")
        if not (isinstance(handle_existence, NamedSemaphore.Flags)):
            raise ValueError("`handle_existence` must be a NamedSemaphore.Flags enum")

        # Save the input parameters
        self._name = "/" + name
        self._unlink_on_delete = unlink_on_delete
        self._unlink_on_signal = unlink_on_signal

        # Check if the semaphore already exists and remove it if flag is set
        if handle_existence == NamedSemaphore.Flags.DELETE_AND_CREATE:
            try:
                self.unlink()
            except FileNotFoundError:
                pass

        if handle_existence == NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS:
            # Force link to an existing semaphore if flag is set
            try:
                self._semaphore_handle = posix_ipc.Semaphore(self.name)
                self._linked_existing_semaphore = True
            except posix_ipc.ExistentialError:
                raise FileNotFoundError(f"Semaphore '{self.name}' does not exist.")
        else:
            # Create the semaphore or link to an existing one based on the flag
            try:
                # O_EXCL flag will fail if the semaphore already exists
                self._semaphore_handle = posix_ipc.Semaphore(
                    self.name, posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=initial_value
                )
                self._linked_existing_semaphore = False
            except posix_ipc.ExistentialError:
                self._semaphore_handle = posix_ipc.Semaphore(
                    self.name, posix_ipc.O_CREAT, initial_value=initial_value
                )
                self._linked_existing_semaphore = True
                if handle_existence == NamedSemaphore.Flags.RAISE_IF_EXISTS:
                    raise FileExistsError(f"Semaphore '{self.name}' already exists.")
            except PermissionError as e:
                raise PermissionError(
                    f"Permission denied when trying to open the semaphore {self.name}."
                ) from e

        # Register the signal handler if needed
        if self.unlink_on_signal:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, self.__signal_handler)

    @property
    def name(self) -> str:
        """
        Return the name of the semaphore.

        :return: The name of the semaphore.
        :rtype: str
        """
        return self._name

    @property
    def linked_existing_semaphore(self) -> bool:
        """
        Return whether the semaphore was linked to an existing semaphore on handle creation.

        :return: True if verifies condition, False otherwise.
        :rtype: bool
        """
        return self._linked_existing_semaphore

    @property
    def unlink_on_delete(self) -> bool:
        """
        Return whether the semaphore will be unlinked when the object is deleted. The default
        behavior is to unlink the semaphore if it was created by this handle. But can be manually
        overridden by setting the `unlink_on_delete` parameter in the constructor.

        :return: True if the semaphore will be unlinked when the object is deleted, False otherwise.
        :rtype: bool
        """
        if self._unlink_on_del is not None:
            return self._unlink_on_del
        return not self.linked_existing_semaphore

    @property
    def unlink_on_signal(self) -> bool:
        """
        Return whether the semaphore will be unlinked when the process receives a SIGINT, SIGTERM,
        or SIGHUP signal. The default behavior is to set up the signal handler if the semaphore was
        created by this handle. But can be manually overridden by setting the `unlink_on_signal`
        parameter in the constructor.

        :return: True if the semaphore will be unlinked when the process receives a signal, False
            otherwise.
        :rtype: bool
        """
        if self._unlink_on_signal is not None:
            return self._unlink_on_signal
        return not self.linked_existing_semaphore

    def acquire(self, blocking: bool = True, timeout: Optional[Real] = None) -> bool:
        """
        Acquire the semaphore.

        :param bool blocking: If True, the method will block until the semaphore is acquired. If False,
            the method will return immediately, regardless of whether the semaphore was acquired.
        :param Real timeout: If provided, the method will block for at most `timeout` seconds. If the
            semaphore is not acquired within this time, the method will return False. If not provided,
            the method will block indefinitely if `blocking` is True.
        :return: True if the semaphore was acquired, False otherwise.
        :rtype: bool
        """
        # Check the input parameters
        if not isinstance(blocking, bool):
            raise ValueError("`blocking` must be a boolean")
        if timeout is not None and not isinstance(timeout, Real):
            raise ValueError("If provided, `timeout` must be a real number")

        try:  # General error handling for corrupted semaphores
            # Non-blocking acquire
            if not blocking:
                if timeout is not None:
                    raise ValueError("Cannot specify a timeout if blocking is False")
                try:
                    self._semaphore_handle.acquire(nowait=True)
                    return True
                except posix_ipc.BusyError:
                    return False
            # Blocking acquire, no timeout
            if timeout is None:
                self._semaphore_handle.acquire()
                return True

            # Blocking acquire with timeout
            try:
                self._semaphore_handle.acquire(timeout=timeout)
                return True
            except posix_ipc.TimeoutError:
                return False
        except (SystemError, OSError) as e:
            self.logger.error(f"Error while acquiring: {e}.", exc_info=True)
            raise

    def release(self, n: int = 1) -> None:
        """
        Release the semaphore.

        :param int n: The number of times to release the semaphore. Default is 1.
        :raises ValueError: If `n` is invalid.
        :raises PermissionError: If the semaphore cannot be released due to permissions.
        :raises OSError: If a system-level error occurs.
        """
        # Check the input parameters
        if not (isinstance(n, int) and n >= 1):
            raise ValueError("`n` must be a positive integer")

        # Release the semaphore
        try:
            for _ in range(n):
                self._semaphore_handle.release()
        except posix_ipc.PermissionError as e:
            raise PermissionError(
                f"Cannot release semaphore {self.name}: Permission denied."
            ) from e
        except OSError as e:
            raise OSError(f"System error occurred while releasing semaphore {self.name}.") from e

    def unlink(self) -> None:
        """
        Unlink the semaphore.

        This method removes the semaphore globally, making it inaccessible by its name.
        Any other processes linked to this semaphore will lose access to it. Use this method
        cautiously in shared environments.

        :raises FileNotFoundError: If the semaphore cannot be unlinked.
        """
        try:
            posix_ipc.unlink_semaphore(self.name)
        except posix_ipc.ExistentialError:
            raise FileNotFoundError(f"Semaphore '{self.name}' does not exist.")

    def __enter__(self) -> Self:
        """
        Enter the semaphore context. Acquires the semaphore.

        :return: The created object.
        :rtype: Self
        """
        self.acquire()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        Exit the semaphore context. Releases the semaphore.
        """
        # Try to release the semaphore
        try:
            self.release()
        except Exception as e:
            self.logger.error(f"Error while releasing semaphore in __exit__: {e}", exc_info=True)

    def __del__(self):
        """
        Destructor for the class. Unlinks the semaphore if it was created by this handle.
        """
        try:
            if self.unlink_on_delete:
                self.__cleanup()
        except Exception as e:
            self.logger.error(f"Error in __del__: {e}", exc_info=True)

    def __cleanup(self):
        """
        Helper method to unlink the semaphore.
        """
        try:
            self.unlink()
        except FileNotFoundError:  # Ignore if the semaphore does not exist
            pass
        except Exception as e:
            self.logger.error(f"Error during semaphore cleanup: {e}", exc_info=True)

    def __signal_handler(self, signum, frame):
        """
        Signal handler for the semaphore. Unlinks the semaphore and exits the process with the
        standard exit code for the signal. Only unlinked if the `unlink_on_delete` parameter is True.
        """
        if self.unlink_on_signal:
            self.__cleanup()
        sys.exit(128 + signum)  # 128+n is the standard exit code for signal n
