"""Use cases for non-streaming app run jobs: open, attach a reader, view, wait and stop."""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, Protocol

from flask import copy_current_request_context

from models.enums import CreatorUserRole
from models.model import AppMode
from services.run_job_contracts import (
    JobFamily,
    JobOwner,
    JobPatch,
    JobRecord,
    JobStatus,
    OpenedJob,
    RunJobStore,
)
from services.run_job_reader import JobStopper, read_to_end

if TYPE_CHECKING:
    from services.app_task_service import WorkflowStopOutcome

WAIT_DEFAULT_SECONDS: Final = 25
WAIT_MAX_SECONDS: Final = 45
POLL_INTERVAL_SECONDS: Final = 1.0
READER_THREAD_NAME: Final = "run-job-reader"
RUN_NOT_CONFIRMED: Final = "run start was not confirmed; it may still execute"
RUN_INTERRUPTED: Final = "run was interrupted"


@dataclass(frozen=True, slots=True, kw_only=True)
class JobOpening:
    owner: JobOwner
    mode: AppMode
    op: str
    idempotency_key: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult:
    status: JobStatus
    outputs: Mapping[str, Any] | None = None
    answer: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageResult:
    answer: str
    error: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class JobView:
    record: JobRecord
    status: JobStatus
    answer: str | None = None
    outputs: Mapping[str, Any] | None = None
    error: str | None = None
    form: Mapping[str, Any] | None = None


class RunResultReader(Protocol):
    def run(self, *, tenant_id: str, run_id: str) -> RunResult | None:
        """Precondition: none. Postcondition: the stored result of the run, or None when it is not stored."""

    def pending_form(self, *, tenant_id: str, app_id: str, run_id: str, mode: AppMode) -> Mapping[str, Any] | None:
        """Precondition: the run is paused. Postcondition: its pending human input form, or None."""

    def message(self, *, app_id: str, message_id: str) -> MessageResult | None:
        """Precondition: none. Postcondition: the saved answer and error of the message, or None when not stored."""


class RunTaskControl(Protocol):
    def stop_workflow_run(
        self,
        *,
        tenant_id: str,
        app_id: str,
        run_id: str,
        app_mode: AppMode,
        caller_role: CreatorUserRole,
        caller_id: str,
    ) -> WorkflowStopOutcome:
        """Precondition: none. Postcondition: the run is asked to stop and the outcome is reported."""

    def stop_message_task(self, *, task_id: str, caller_role: CreatorUserRole, caller_id: str) -> bool:
        """Precondition: none. Postcondition: the task is asked to stop."""


class JobNotFoundError(Exception):
    """The job does not exist or belongs to another owner."""


class JobCapacityError(Exception):
    """The background run cap of this process is reached."""


def _has_run_id(record: JobRecord) -> bool:
    """Report whether the record already links to a run row.

    Precondition: none.
    Postcondition: true when run_id is set.
    """
    return record.run_id is not None


def _never_confirmed(record: JobRecord) -> bool:
    """Report that a message job has no state that confirms its reader besides the alive key.

    Precondition: none.
    Postcondition: always false.
    """
    return False


_CONFIRMED_BY_FAMILY: Final[Mapping[JobFamily, Callable[[JobRecord], bool]]] = {
    JobFamily.GRAPH: _has_run_id,
    JobFamily.MESSAGE: _never_confirmed,
}


def fail_if_reader_dead(store: RunJobStore, record: JobRecord, error: str) -> JobRecord:
    """Fail a job whose reader has ended before the job reached a final status.

    Precondition: record is the stored state of the job. The reader refreshes the alive key on a timer for as long as
    it runs, frames or not, so the key lapses only when the reader has ended.
    Postcondition: the record unchanged when it is final or its alive key is live. Otherwise the job is read again
    so a change made since record was read is not overwritten: the fresh record is returned untouched when it is
    final, a GRAPH job now has a run id, or the alive key is live again; record is returned when the job is gone
    from the store. In every other case the job is saved as FAILED with the given error through store.patch and
    the fresh record is returned as FAILED.
    """
    if record.status.is_final:
        return record
    if store.is_alive(record.id):
        return record
    fresh = store.get(record.id)
    if fresh is None:
        return record
    if fresh.status.is_final:
        return fresh
    if _CONFIRMED_BY_FAMILY[fresh.family](fresh):
        return fresh
    if store.is_alive(fresh.id):
        return fresh
    store.patch(fresh.id, JobPatch(status=JobStatus.FAILED, error=error))
    return replace(fresh, status=JobStatus.FAILED, error=error)


class _Counter:
    """A thread-safe bounded counter of held slots."""

    def __init__(self, limit: int) -> None:
        """Create a counter.

        Precondition: limit is positive.
        Postcondition: no slot is held.
        """
        self._limit = limit
        self._held = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Take one slot.

        Precondition: none.
        Postcondition: true and one more slot is held when fewer than the limit were held; false otherwise.
        """
        with self._lock:
            if self._held >= self._limit:
                return False
            self._held += 1
            return True

    def release(self) -> None:
        """Give back one slot.

        Precondition: none.
        Postcondition: one fewer slot is held; the count never goes below zero.
        """
        with self._lock:
            self._held = max(0, self._held - 1)


def _stop_graph_job(tasks: RunTaskControl, record: JobRecord) -> None:
    """Stop the workflow run of a graph job when its run id is known.

    Precondition: record is a GRAPH job.
    Postcondition: the run is asked to stop; nothing happens while run_id is unset.
    """
    if record.run_id is None:
        return
    tasks.stop_workflow_run(
        tenant_id=record.owner.tenant_id,
        app_id=record.owner.app_id,
        run_id=record.run_id,
        app_mode=record.mode,
        caller_role=record.owner.caller_role,
        caller_id=record.owner.caller_id,
    )


def _stop_message_job(tasks: RunTaskControl, record: JobRecord) -> None:
    """Stop the task of a message job when its task id is known.

    Precondition: record is a MESSAGE job.
    Postcondition: the task is asked to stop; nothing happens while task_id is unset.
    """
    if record.task_id is None:
        return
    tasks.stop_message_task(
        task_id=record.task_id,
        caller_role=record.owner.caller_role,
        caller_id=record.owner.caller_id,
    )


_STOPS_BY_FAMILY: Final[Mapping[JobFamily, Callable[[RunTaskControl, JobRecord], None]]] = {
    JobFamily.GRAPH: _stop_graph_job,
    JobFamily.MESSAGE: _stop_message_job,
}


class TaskControlStopper:
    """Stops the run behind a job record through task control; serves the reader and RunJobService.stop."""

    def __init__(self, tasks: RunTaskControl) -> None:
        """Bind the task control.

        Precondition: none.
        Postcondition: stop sends stops through tasks.
        """
        self._tasks = tasks

    def stop(self, record: JobRecord) -> None:
        """Ask the run behind the record to stop.

        Precondition: none.
        Postcondition: the stop for the record's family was sent when the id it needs is known; otherwise nothing
        happened.
        """
        _STOPS_BY_FAMILY[record.family](self._tasks, record)


class RunJobService:
    """Opens jobs, attaches their readers, and answers view, wait and stop for the owner of a job.

    Precondition: built once per process with a store, result reader and task control.
    Postcondition: background runs and waiters of this process stay within their caps.
    """

    def __init__(
        self,
        *,
        store: RunJobStore,
        results: RunResultReader,
        tasks: RunTaskControl,
        max_background_runs: int,
        max_waiters: int,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the service.

        Precondition: both caps are positive; sleep and monotonic are consistent with each other.
        Postcondition: no slot, waiter or reservation is held.
        """
        self._store = store
        self._results = results
        self._stopper: JobStopper = TaskControlStopper(tasks)
        self._slots = _Counter(max_background_runs)
        self._waiters = _Counter(max_waiters)
        self._sleep = sleep
        self._monotonic = monotonic
        self._reserved: set[str] = set()
        self._reserved_lock = threading.Lock()

    def open(self, opening: JobOpening) -> OpenedJob:
        """Create a job, or return the earlier one for the same idempotency key.

        Precondition: opening.mode supports run jobs.
        Postcondition: a created job holds one background slot until attach hands it to the reader thread or
        discard releases it. An earlier job holds none. When the store raises, no slot is held. When the cap is
        reached the new job is removed and JobCapacityError is raised.
        """
        record = JobRecord(
            id=str(uuid.uuid4()),
            owner=opening.owner,
            mode=opening.mode,
            op=opening.op,
            status=JobStatus.RUNNING,
        )
        # Store first so a repeated key at the cap still returns its earlier job.
        opened = self._store.open(record, idempotency_key=opening.idempotency_key)
        if not opened.created:
            return opened
        if self._slots.try_acquire():
            self._reserve(opened.record.id)
            return opened
        self._store.discard(opened.record.id, idempotency_key=opening.idempotency_key)
        raise JobCapacityError("Too many background runs are active")

    def _reserve(self, job_id: str) -> None:
        """Remember that job_id holds a slot taken by open.

        Precondition: the slot was just acquired.
        Postcondition: job_id is reserved.
        """
        with self._reserved_lock:
            self._reserved.add(job_id)

    def _unreserve(self, job_id: str) -> bool:
        """Forget a reservation.

        Precondition: none.
        Postcondition: true when job_id was reserved by this process's open (now forgotten); false otherwise.
        """
        with self._reserved_lock:
            if job_id not in self._reserved:
                return False
            self._reserved.remove(job_id)
            return True

    def attach(self, job_id: str, frames: Iterable[str]) -> threading.Thread:
        """Read the run's frames on a daemon thread.

        Precondition: called inside a request context, after this process's open created the job (created=True)
        and reserved its slot; attach is called at most once per job.
        Postcondition: the thread owns the slot and releases it when the stream ends; if the thread cannot start,
        the slot is released and the error propagates. Raises ValueError, taking no slot, for a job this
        process's open did not create.
        """
        if not self._unreserve(job_id):
            raise ValueError(f"Job {job_id} holds no reserved background slot")
        context = contextvars.copy_context()

        def reader_with_context() -> None:
            """Read the stream under the copied contexts.

            Precondition: runs on the reader thread.
            Postcondition: the stream was read to its end and the slot is released.
            """
            try:
                context.run(
                    read_to_end,
                    job_id=job_id,
                    frames=frames,
                    store=self._store,
                    stopper=self._stopper,
                )
            finally:
                self._slots.release()

        try:
            thread = threading.Thread(
                target=copy_current_request_context(reader_with_context),
                name=READER_THREAD_NAME,
                daemon=True,
            )
            thread.start()
        except BaseException:
            self._slots.release()
            raise
        return thread

    def discard(self, opened: OpenedJob, *, idempotency_key: str | None) -> None:
        """Remove a job whose run never started.

        Precondition: opened came from open and attach was not called for it.
        Postcondition: the slot reserved by this process's open is released even when the store raises (the error
        propagates); otherwise the job is gone.
        """
        try:
            self._store.discard(opened.record.id, idempotency_key=idempotency_key)
        finally:
            self._release_reserved_slot(opened)

    def _release_reserved_slot(self, opened: OpenedJob) -> None:
        """Give back the slot that open reserved for a job.

        Precondition: none.
        Postcondition: the slot is released only when opened was created by this process's open and still reserved.
        """
        if not opened.created:
            return
        if self._unreserve(opened.record.id):
            self._slots.release()

    def view(self, owner: JobOwner, job_id: str) -> JobView:
        """Describe a job's current state.

        Precondition: none.
        Postcondition: a view of the job. A GRAPH job with a stored run takes status, outputs and error from the
        run, and its form from the pending form while paused (None otherwise). A GRAPH job without a run id whose
        reader has ended and which is not final is FAILED with RUN_NOT_CONFIRMED, and that is saved; once saved it
        is final for good, even if a run id and a run row appear later. Without a
        visible run row the record's own status, error and form are used. A MESSAGE job whose reader has ended
        and which is not final is FAILED with RUN_INTERRUPTED, and that is saved. For a job that is final, with
        a message id and a visible message row, the answer is the message's answer; a MESSAGE job also takes the
        message's error, with the record's error as the fallback. Raises JobNotFoundError when the job is missing
        or owned by someone else.
        """
        record = self._store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        if record.owner != owner:
            raise JobNotFoundError(job_id)
        return self._view_of(record)

    def _view_of(self, record: JobRecord) -> JobView:
        """Build the view of a record.

        Precondition: record is stored.
        Postcondition: see view.
        """
        return _VIEWS_BY_FAMILY[record.family](self, record)

    def _graph_view(self, record: JobRecord) -> JobView:
        """Build the view of a GRAPH job.

        Precondition: record is a GRAPH job.
        Postcondition: see view.
        """
        if record.status.is_final and record.error == RUN_NOT_CONFIRMED:
            # Already reported final: a run that starts late is not linked to the job.
            return _record_view(record)
        if record.run_id is None:
            unconfirmed = fail_if_reader_dead(self._store, record, RUN_NOT_CONFIRMED)
            return _record_view(unconfirmed)
        run = self._results.run(tenant_id=record.owner.tenant_id, run_id=record.run_id)
        if run is None:
            return _record_view(record)
        form = None
        if run.status is JobStatus.PAUSED:
            form = self._results.pending_form(
                tenant_id=record.owner.tenant_id,
                app_id=record.owner.app_id,
                run_id=record.run_id,
                mode=record.mode,
            )
        view = JobView(record=record, status=run.status, outputs=run.outputs, error=run.error, form=form)
        if not run.status.is_final:
            return view
        message = self._saved_message(record)
        if message is None:
            return view
        return replace(view, answer=message.answer)

    def _message_view(self, record: JobRecord) -> JobView:
        """Build the view of a MESSAGE job.

        Precondition: record is a MESSAGE job.
        Postcondition: see view.
        """
        checked = fail_if_reader_dead(self._store, record, RUN_INTERRUPTED)
        view = _record_view(checked)
        if not checked.status.is_final:
            return view
        message = self._saved_message(checked)
        if message is None:
            return view
        error = message.error if message.error is not None else checked.error
        return replace(view, answer=message.answer, error=error)

    def _saved_message(self, record: JobRecord) -> MessageResult | None:
        """Read the saved message of a finished job.

        Precondition: the job's status is final.
        Postcondition: the message row's answer and error; None when the record has no message id or the row is
        not visible.
        """
        if record.message_id is None:
            return None
        return self._results.message(app_id=record.owner.app_id, message_id=record.message_id)

    def wait(self, owner: JobOwner, job_id: str, *, wait_seconds: int) -> JobView:
        """Wait for a job to finish or pause.

        Precondition: wait_seconds is not negative.
        Postcondition: returns once the job is final or paused with a waiting form, the clamped wait time is used
        up, or at once for wait_seconds of 0 or when max_waiters calls already wait in this process. Raises
        JobNotFoundError as view does.
        """
        view = self.view(owner, job_id)
        seconds = min(wait_seconds, WAIT_MAX_SECONDS)
        if seconds <= 0:
            return view
        if not self._waiters.try_acquire():
            return view
        try:
            deadline = self._monotonic() + seconds
            remaining = float(seconds)
            while not _settled(view) and remaining > 0:
                self._sleep(min(POLL_INTERVAL_SECONDS, remaining))
                view = self.view(owner, job_id)
                remaining = deadline - self._monotonic()
        finally:
            self._waiters.release()
        return view

    def stop(self, owner: JobOwner, job_id: str) -> JobView:
        """Ask a job to stop.

        Precondition: none.
        Postcondition: for a job that is not final, stop_requested is stored and the run is stopped now when its id
        is known, otherwise the reader sends the stop once the id arrives; a final job is left untouched and
        nothing is sent. Returns the fresh view. Raises JobNotFoundError when the job is missing or owned by
        someone else.
        """
        record = self._store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        if record.owner != owner:
            raise JobNotFoundError(job_id)
        view = self._view_of(record)
        if view.status.is_final:
            return view
        self._store.patch(job_id, JobPatch(stop_requested=True))
        # Re-read so ids stored by the reader since the first read are not missed.
        stopping = replace(self._store.get(job_id) or record, stop_requested=True)
        self._stopper.stop(stopping)
        return self._view_of(stopping)


def _record_view(record: JobRecord) -> JobView:
    """Build a view from the record's own fields.

    Precondition: none.
    Postcondition: status, error and form are the record's; answer and outputs are None.
    """
    return JobView(record=record, status=record.status, error=record.error, form=record.form)


_VIEWS_BY_FAMILY: Final[Mapping[JobFamily, Callable[[RunJobService, JobRecord], JobView]]] = {
    JobFamily.GRAPH: RunJobService._graph_view,
    JobFamily.MESSAGE: RunJobService._message_view,
}


def _settled(view: JobView) -> bool:
    """Report whether waiting for the job is pointless.

    Precondition: none.
    Postcondition: true when the view's status is final, or paused with a form waiting for input. A paused job
    without a form is being resumed, so it is not settled.
    """
    if view.status.is_final:
        return True
    return view.status is JobStatus.PAUSED and view.form is not None
