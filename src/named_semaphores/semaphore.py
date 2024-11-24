"""
semaphore.py

This module contains a class to handle a POSIX IPC named semaphore.

:author: 2024 Joaquin Jimenez
"""

from enum import Enum
from numbers import Real
from typing import Optional

import posix_ipc
from typing_extensions import Self

from named_semaphores.logging import LoggingMixin


class NamedSemaphore(LoggingMixin):
    """
    Class to handle a POSIX-IPC named semaphore.

    This class provides a Pythonic interface to POSIX named semaphores. It supports multi-process
    environments and relies on the underlying thread-safe POSIX IPC implementation. After creation,
    the semaphore handle is primarily read-only, ensuring thread safety for typical usage.

    This semaphores are supported on Linux, macOS and Windows + Cygwin ≥ 1.7. Note that some features
    such as timeouts are not supported on all platforms (e.g. macOS).

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

        :raises ValueError: If the input parameters are invalid.
        :raises PermissionError: If the semaphore cannot be created (or deleted with
            `DELETE_AND_CREATE`) due to permissions.
        :raises FileExistsError: If the semaphore already exists and could not be removed after
            setting `handle_existence` to `RAISE_IF_EXISTS`.
        :raises FileNotFoundError: If the semaphore could not be found after setting
            `handle_existence` to `RAISE_IF_NOT_EXISTS`.
        """
        # Save the input parameters
        self._name = "/" + name.removeprefix("/") if isinstance(name, str) else ""
        self._unlink_on_delete = unlink_on_delete
        self._linked_existing_semaphore = None

        # Initialize the logger
        LoggingMixin.__init__(self, self._name[1:])

        # Check the input parameters
        if not self.name[1:] or not all(c.isalnum() or c in ("-", "_") for c in self.name[1:]):
            raise ValueError(
                "`name` must be a non-empty string with characters '-', '_' or alphanumeric. "
                f"Got: {name}"
            )
        if not (
            isinstance(initial_value, int) and 0 <= initial_value <= posix_ipc.SEMAPHORE_VALUE_MAX
        ):
            raise ValueError(
                f"`initial_value` must be a non-negative integer less than {posix_ipc.SEMAPHORE_VALUE_MAX}"
            )
        if not (isinstance(handle_existence, NamedSemaphore.Flags)):
            raise ValueError("`handle_existence` must be a NamedSemaphore.Flags enum")

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
            return

        # Create the semaphore or link to an existing one based on the flag
        try:
            try:
                # O_CREX flag will fail with ExistentialError if the semaphore already exists
                self._semaphore_handle = posix_ipc.Semaphore(
                    self.name, posix_ipc.O_CREX, initial_value=initial_value
                )
                self._linked_existing_semaphore = False
            except posix_ipc.ExistentialError:  # Try to link
                # Link to an existing semaphore
                self._semaphore_handle = posix_ipc.Semaphore(
                    self.name, posix_ipc.O_CREAT, initial_value=initial_value
                )
                self._linked_existing_semaphore = True
                if handle_existence == NamedSemaphore.Flags.RAISE_IF_EXISTS:
                    raise FileExistsError(f"Semaphore '{self.name}' already exists.")
        except posix_ipc.PermissionsError as e:
            raise PermissionError(f"Permission denied creating semaphore {self.name}.") from e

    @property
    def name(self) -> str:
        """
        Return the name of the semaphore.

        :return: The name of the semaphore.
        :rtype: str
        """
        return self._name

    @property
    def linked_existing_semaphore(self) -> Optional[bool]:
        """
        Return whether the semaphore was linked to an existing semaphore on handle creation.

        :return: True if verifies condition, False otherwise. None if not yet verified.
        :rtype: Optional[bool]
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
        if self._unlink_on_delete is not None:
            return self._unlink_on_delete
        return self._linked_existing_semaphore is False

    @property
    def value(self) -> int:
        """
        Return the current value of the semaphore. Not possible on macOS.

        :return: The current value of the semaphore.
        :rtype: int
        """
        if not posix_ipc.SEMAPHORE_VALUE_SUPPORTED:
            raise NotImplementedError("Operation is not supported on this platform")
        return self._semaphore_handle.value

    def acquire(self, blocking: bool = True, timeout: Optional[Real] = None) -> bool:
        """
        Acquire the semaphore.

        :param bool blocking: If True, the method will block until the semaphore is acquired. If False,
            the method will return immediately, regardless of whether the semaphore was acquired.
        :param Real timeout: If provided, the method will block for at most `timeout` seconds. If the
            semaphore is not acquired within this time, the method will return False. If not provided,
            the method will block indefinitely if `blocking` is True. Not supported on macOS.
        :return: True if the semaphore was acquired, False otherwise.
        :rtype: bool
        :raises ValueError: If the input parameters are invalid.
        :raises NotImplementedError: If the platform does not support timeout and a timeout is provided.
        """
        # Check the input parameters
        if not isinstance(blocking, bool):
            raise ValueError("`blocking` must be a boolean")
        if timeout is not None and (not isinstance(timeout, Real) or timeout < 0):
            raise ValueError("If provided, `timeout` must be a positive real number")

        acquire_kwargs = {}  # Setting for the default blocking acquire
        # Non-blocking acquire
        if not blocking:
            acquire_kwargs["timeout"] = 0
            if timeout is not None:
                raise ValueError("Cannot specify a timeout if blocking is False")
        # Blocking acquire with timeout
        elif timeout is not None:
            acquire_kwargs["timeout"] = timeout
            if not posix_ipc.SEMAPHORE_TIMEOUT_SUPPORTED:
                raise NotImplementedError("Timeouts are not supported on this platform")

        # Blocking acquire with timeout
        try:
            self._semaphore_handle.acquire(**acquire_kwargs)
            return True
        except posix_ipc.BusyError:
            return False

    def release(self, n: int = 1) -> None:
        """
        Release the semaphore.

        :param int n: The number of times to release the semaphore. Default is 1.
        :raises ValueError: If `n` is invalid.
        """
        # Check the input parameters
        if not (isinstance(n, int) and n >= 1):
            raise ValueError("`n` must be a positive integer")

        # Release the semaphore
        for _ in range(n):
            self._semaphore_handle.release()

    def unlink(self) -> None:
        """
        Unlink the semaphore.

        This method removes the semaphore globally, making it inaccessible by its name.
        Any other processes linked to this semaphore will lose access to it. Use this method
        cautiously in shared environments.

        :raises FileNotFoundError: If the semaphore cannot be unlinked due to not existing.
        :raises PermissionError: If the semaphore cannot be unlinked due to permissions.
        """
        try:
            posix_ipc.unlink_semaphore(self.name)
        except posix_ipc.ExistentialError:
            raise FileNotFoundError(f"Semaphore '{self.name}' does not exist.")
        except posix_ipc.PermissionsError as e:
            raise PermissionError(f"Permission denied unlinking semaphore {self.name}.") from e

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
        self.release()

    def __del__(self):
        """
        Destructor for the class. Unlinks the semaphore if it was created by this handle.
        """
        # Close the semaphore handle
        if getattr(self, "_semaphore_handle", None) is not None:
            try:
                self._semaphore_handle.close()
            except posix_ipc.ExistentialError:
                pass

        # Unlink the semaphore if it was created by this handle
        if not self.unlink_on_delete:
            return
        try:
            # Unlink the semaphore
            self.unlink()
        except FileNotFoundError:  # Ignore if the semaphore does not exist
            pass
        except PermissionError:
            self.logger.warning("Permission denied unlinking semaphore during cleanup.")
