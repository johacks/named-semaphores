import pytest
import posix_ipc
from easy_posix_ipc.semaphore import NamedSemaphore
import random
import multiprocessing as mp
import os
import signal
from unittest.mock import patch


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


class TestNamedSemaphore:
    def test_init_basic(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        assert sem.name == f"/{semaphore_name}"
        assert sem.linked_existing_semaphore is False

    def test_init_invalid_name(self):
        with pytest.raises(ValueError):
            NamedSemaphore("")
        with pytest.raises(ValueError):
            NamedSemaphore("test@semaphore")

    def test_init_invalid_initial_value(self, semaphore_name):
        with pytest.raises(ValueError):
            NamedSemaphore(semaphore_name, initial_value=-1)
        with pytest.raises(ValueError):
            NamedSemaphore(semaphore_name, initial_value="1")

    def test_init_invalid_handle_existence(self, semaphore_name):
        with pytest.raises(ValueError):
            NamedSemaphore(semaphore_name, handle_existence=100)
        with pytest.raises(ValueError):
            NamedSemaphore(semaphore_name, handle_existence="RAISE_IF_EXISTS")

    def test_raise_if_exists(self, semaphore_name):
        # First creation should succeed
        sem1 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)
        assert sem1.linked_existing_semaphore is False

        # Second creation should fail
        with pytest.raises(FileExistsError):
            NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)

    def test_raise_if_not_exists_when_not_exists(self, semaphore_name):
        # Should fail when semaphore doesn't exist
        with pytest.raises(FileNotFoundError):
            NamedSemaphore(
                semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
            )

    def test_raise_if_not_exists_when_exists(self, semaphore_name):
        # Create first semaphore
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        assert sem.linked_existing_semaphore is False

        # Successful to existing semaphore
        sem_link = NamedSemaphore(
            semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
        )
        assert sem_link.linked_existing_semaphore is True

    def test_link_or_create(self, semaphore_name):
        # First creation
        sem1 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        assert sem1.linked_existing_semaphore is False

        # Second should link
        sem2 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        assert sem2.linked_existing_semaphore is True

    def test_delete_and_create(self, semaphore_name):
        # Create first semaphore
        NamedSemaphore(
            semaphore_name,
            handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
            unlink_on_delete=False,  # Don't unlink the semaphore on garbage collection
        )

        # Delete and create new one
        sem = NamedSemaphore(
            semaphore_name, handle_existence=NamedSemaphore.Flags.DELETE_AND_CREATE
        )
        assert sem.linked_existing_semaphore is False

    def test_delete_and_create_no_fail_if_not_exists(self, semaphore_name):
        # Delete and create new one
        sem = NamedSemaphore(
            semaphore_name, handle_existence=NamedSemaphore.Flags.DELETE_AND_CREATE
        )
        assert sem.linked_existing_semaphore is False

    def test_value(self, semaphore_name):
        sem = NamedSemaphore(
            semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE, initial_value=2
        )
        assert sem.value == 2
        sem.acquire()
        assert sem.value == 1
        sem.release()
        assert sem.value == 2

    def test_value_bad_os(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        with patch("posix_ipc.SEMAPHORE_VALUE_SUPPORTED", False):
            with pytest.raises(NotImplementedError):
                sem.value

    def test_acquire_bad_timeout(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        with pytest.raises(ValueError):
            sem.acquire(blocking=True, timeout=-1)
        with pytest.raises(ValueError):
            sem.acquire(blocking=True, timeout="1")
        with pytest.raises(ValueError):  # Timeout cannot be provided for non-blocking acquire
            sem.acquire(blocking=False, timeout=1)

    def test_acquire_bad_blocking(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        with pytest.raises(ValueError):
            sem.acquire(blocking=100)
        with pytest.raises(ValueError):
            sem.acquire(blocking="True")

    def test_acquire_release(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        assert sem.acquire(blocking=True) is True
        sem.release()

    def test_acquire_timeout(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        # First acquire should succeed
        assert sem.acquire(blocking=True) is True

        # Second acquire should timeout
        assert sem.acquire(blocking=True, timeout=0.1) is False

    def test_acquire_timeout_bad_os(self, semaphore_name):
        with patch("posix_ipc.SEMAPHORE_TIMEOUT_SUPPORTED", False):
            sem = NamedSemaphore(
                semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE
            )

            # First acquire should succeed
            assert sem.acquire(blocking=True) is True

            # Second acquire with timeout should fail with NotImplementedError
            with pytest.raises(NotImplementedError):
                sem.acquire(blocking=True, timeout=0.1)

    def test_acquire_non_blocking(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        # First non-blocking acquire should succeed
        assert sem.acquire(blocking=False) is True

        # Second non-blocking acquire should fail
        assert sem.acquire(blocking=False) is False

    def test_context_manager(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        with sem:
            # Semaphore should be acquired here
            assert sem.acquire(blocking=False) is False

        # Semaphore should be released here
        assert sem.acquire(blocking=False) is True

    def test_release_bad_n(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)

        with pytest.raises(ValueError):
            sem.release(n=-1)
        with pytest.raises(ValueError):
            sem.release(n="1")

    def test_multiple_release(self, semaphore_name):
        sem = NamedSemaphore(
            semaphore_name, initial_value=0, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE
        )

        sem.release(n=3)

        # Should be able to acquire 3 times
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is True
        assert sem.acquire(blocking=False) is False

    def test_unlink(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        sem.unlink()

        # Should raise when trying to link to non-existent semaphore
        with pytest.raises(FileNotFoundError):
            NamedSemaphore(
                semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
            )

    def test_unlink_on_delete_auto_mode(self, semaphore_name):
        sem = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE)
        assert sem.unlink_on_delete is True
        sem.__del__()
        with pytest.raises(posix_ipc.ExistentialError):
            posix_ipc.unlink_semaphore(f"/{semaphore_name}")

    def test_unlink_on_delete_explicit_mode_to_false(self, semaphore_name):
        sem = NamedSemaphore(
            semaphore_name,
            handle_existence=NamedSemaphore.Flags.LINK_OR_CREATE,
            unlink_on_delete=False,
        )
        assert sem.unlink_on_delete is False
        sem.__del__()
        posix_ipc.unlink_semaphore(f"/{semaphore_name}")

    def test_unlink_on_sigint(self, semaphore_name):
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

    def test_unlink_on_signal_unhandled_signal(self, semaphore_name):
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
