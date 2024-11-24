import pytest
import posix_ipc
from named_semaphores.semaphore import NamedSemaphore
import random
import multiprocessing as mp
import os
import signal
from unittest.mock import patch
from contextlib import contextmanager


@contextmanager
def dropping_root_privileges():
    # Drop privileges to whatever UID and GID are provided in the environment
    uid = int(os.environ.get("UID", 1000))
    gid = int(os.environ.get("GID", 1000))
    os.setegid(gid)
    os.seteuid(uid)

    yield

    # Restore root privileges
    os.seteuid(0)
    os.setegid(0)


@pytest.fixture
def require_root():
    if os.getuid() != 0:
        pytest.skip("Test requires root privileges. Re-run with sudo -E env PATH=$PATH pytest ...")


@pytest.fixture
def semaphore_name():
    # It is better if each unit test has a unique semaphore name, for isolation purposes
    return "test_semaphore_" + str(random.randint(0, 2**24))


# Helper function to create a semaphore in a separate process and block it
def create_semaphore_task(semaphore_name, event):
    sem = NamedSemaphore(
        semaphore_name, initial_value=0, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS
    )
    # Signal the main process that semaphore is created
    event.set()
    sem.acquire()
    while True:
        pass


@pytest.fixture(autouse=True)
def cleanup_semaphore(semaphore_name):
    # Cleanup before test
    try:
        posix_ipc.unlink_semaphore(f"/{semaphore_name}")
    except posix_ipc.ExistentialError:
        pass

    yield

    # Cleanup after test
    try:
        posix_ipc.unlink_semaphore(f"/{semaphore_name}")
    except posix_ipc.ExistentialError:
        pass


def test_init_basic(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    assert sem.name == f"/{semaphore_name}"
    assert sem.linked_existing_semaphore is False


def test_init_invalid_name():
    with pytest.raises(ValueError):
        NamedSemaphore("")
    with pytest.raises(ValueError):
        NamedSemaphore("test@semaphore")


def test_init_invalid_initial_value(semaphore_name):
    with pytest.raises(ValueError):
        NamedSemaphore(semaphore_name, initial_value=-1)
    with pytest.raises(ValueError):
        NamedSemaphore(semaphore_name, initial_value="1")


def test_init_invalid_handle_existence(semaphore_name):
    with pytest.raises(ValueError):
        NamedSemaphore(semaphore_name, handle_existence=100)
    with pytest.raises(ValueError):
        NamedSemaphore(semaphore_name, handle_existence="RAISE_IF_EXISTS")


def test_raise_if_exists(semaphore_name):
    # First creation should succeed
    sem1 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)
    assert sem1.linked_existing_semaphore is False

    # Second creation should fail
    with pytest.raises(FileExistsError):
        NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)


def test_raise_if_not_exists_when_not_exists(semaphore_name):
    # Should fail when semaphore doesn't exist
    with pytest.raises(FileNotFoundError):
        NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS)


def test_raise_if_not_exists_when_exists(semaphore_name):
    # Create first semaphore
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    assert sem.linked_existing_semaphore is False

    # Successful to existing semaphore
    sem_link = NamedSemaphore(
        semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
    )
    assert sem_link.linked_existing_semaphore is True


def test_link_or_create(semaphore_name):
    # First creation
    sem1 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    assert sem1.linked_existing_semaphore is False

    # Second should link
    sem2 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    assert sem2.linked_existing_semaphore is True


def test_unlink_and_create(semaphore_name):
    # Create first semaphore
    NamedSemaphore(
        semaphore_name,
        handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
        unlink_on_delete=False,  # Don't unlink the semaphore on garbage collection
    )

    # Delete and create new one
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.UNLINK_AND_CREATE)
    assert sem.linked_existing_semaphore is False


def test_unlink_and_create_no_fail_if_not_exists(semaphore_name):
    # Delete and create new one
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.UNLINK_AND_CREATE)
    assert sem.linked_existing_semaphore is False


def test_link_bad_permissions(semaphore_name, require_root):
    # Create semaphore with no permissions
    NamedSemaphore(
        semaphore_name,
        handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
        unlink_on_delete=False,
    )

    # Should fail to link to semaphore
    with dropping_root_privileges():
        with pytest.raises(PermissionError):
            NamedSemaphore(
                semaphore_name,
                handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
                unlink_on_delete=False,
            )


def test_value(semaphore_name):
    sem = NamedSemaphore(
        semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE, initial_value=2
    )
    assert sem.value == 2
    sem.acquire()
    assert sem.value == 1
    sem.release()
    assert sem.value == 2


def test_value_bad_os(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    with patch("posix_ipc.SEMAPHORE_VALUE_SUPPORTED", False):
        with pytest.raises(NotImplementedError):
            sem.value


def test_acquire_bad_timeout(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    with pytest.raises(ValueError):
        sem.acquire(blocking=True, timeout=-1)
    with pytest.raises(ValueError):
        sem.acquire(blocking=True, timeout="1")
    with pytest.raises(ValueError):  # Timeout cannot be provided for non-blocking acquire
        sem.acquire(blocking=False, timeout=1)


def test_acquire_bad_blocking(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    with pytest.raises(ValueError):
        sem.acquire(blocking=100)
    with pytest.raises(ValueError):
        sem.acquire(blocking="True")


def test_acquire_release(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    assert sem.acquire(blocking=True) is True
    sem.release()


def test_acquire_timeout(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    # First acquire should succeed
    assert sem.acquire(blocking=True) is True

    # Second acquire should timeout
    assert sem.acquire(blocking=True, timeout=0.1) is False


def test_acquire_timeout_bad_os(semaphore_name):
    with patch("posix_ipc.SEMAPHORE_TIMEOUT_SUPPORTED", False):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        # First acquire should succeed
        assert sem.acquire(blocking=True) is True

        # Second acquire with timeout should fail with NotImplementedError
        with pytest.raises(NotImplementedError):
            sem.acquire(blocking=True, timeout=0.1)


def test_acquire_non_blocking(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    # First non-blocking acquire should succeed
    assert sem.acquire(blocking=False) is True

    # Second non-blocking acquire should fail
    assert sem.acquire(blocking=False) is False


def test_context_manager(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    with sem:
        # Semaphore should be acquired here
        assert sem.acquire(blocking=False) is False

    # Semaphore should be released here
    assert sem.acquire(blocking=False) is True


def test_release_bad_n(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    with pytest.raises(ValueError):
        sem.release(n=-1)
    with pytest.raises(ValueError):
        sem.release(n="1")


def test_multiple_release(semaphore_name):
    sem = NamedSemaphore(
        semaphore_name, initial_value=0, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE
    )

    sem.release(n=3)

    # Should be able to acquire 3 times
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is True
    assert sem.acquire(blocking=False) is False


def test_unlink(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    sem.unlink()

    # Should raise when trying to link to non-existent semaphore
    with pytest.raises(FileNotFoundError):
        NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS)


def test_unlink_bad_permissions(semaphore_name, require_root):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

    with dropping_root_privileges():
        # Should fail to unlink semaphore
        with pytest.raises(PermissionError):
            sem.unlink()
        # Fails with warning, but should not raise
        sem.__del__()


def test_unlink_on_delete_auto_mode(semaphore_name):
    sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
    assert sem.unlink_on_delete is True
    sem.__del__()
    with pytest.raises(posix_ipc.ExistentialError):
        posix_ipc.unlink_semaphore(f"/{semaphore_name}")


def test_unlink_on_delete_explicit_mode_to_false(semaphore_name):
    sem = NamedSemaphore(
        semaphore_name,
        handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
        unlink_on_delete=False,
    )
    assert sem.unlink_on_delete is False
    sem.__del__()
    posix_ipc.unlink_semaphore(f"/{semaphore_name}")


def test_unlink_on_sigint(semaphore_name):
    create_event = mp.Event()
    process = mp.Process(
        target=create_semaphore_task, args=(semaphore_name, create_event), daemon=True
    )
    process.start()
    create_event.wait()
    os.kill(process.pid, signal.SIGINT)
    process.join()

    # Should result in non-zero exit code after KeyboardInterrupt is raised
    assert process.exitcode == 1

    # As SIGINT is handled with normal exit flow, semaphore should be unlinked during cleanup
    with pytest.raises(posix_ipc.ExistentialError):
        posix_ipc.unlink_semaphore(f"/{semaphore_name}")


def test_unlink_on_signal_unhandled_signal(semaphore_name):
    create_event = mp.Event()
    process = mp.Process(
        target=create_semaphore_task, args=(semaphore_name, create_event), daemon=True
    )
    process.start()
    create_event.wait()
    os.kill(process.pid, signal.SIGTERM)
    process.join()
    assert process.exitcode != 0

    # Semaphore should not be unlinked as SIGTERM is not handled
    posix_ipc.unlink_semaphore(f"/{semaphore_name}")
