import pytest
import posix_ipc
from easy_posix_ipc.semaphore import NamedSemaphore
import random


@pytest.fixture
def semaphore_name():
    # It is better if each unit test has a unique semaphore name, for isolation purposes
    return "test_semaphore_" + str(random.randint(0, 2**24))


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

    def test_raise_if_exists(self, semaphore_name):
        # First creation should succeed
        sem1 = NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)
        assert sem1.linked_existing_semaphore is False

        # Second creation should fail
        with pytest.raises(FileExistsError):
            NamedSemaphore(semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_EXISTS)

    def test_raise_if_not_exists(self, semaphore_name):
        # Should fail when semaphore doesn't exist
        with pytest.raises(FileNotFoundError):
            NamedSemaphore(
                semaphore_name, handle_existence=NamedSemaphore.Flags.RAISE_IF_NOT_EXISTS
            )

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
