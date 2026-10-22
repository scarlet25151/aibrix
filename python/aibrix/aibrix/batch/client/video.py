# Copyright 2026 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Videos lifecycle; persistence and cancellation are supplied by the caller."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, BinaryIO, Callable, TypeVar
from urllib.parse import quote, urljoin

import httpx

from aibrix.batch.client.channel import _is_retryable_status, _response_body
from aibrix.batch.client.errors import InferenceError, InferenceErrorCode
from aibrix.logger import init_logger
from aibrix.metadata.core import HTTPXClientWrapper

if TYPE_CHECKING:
    from aibrix.batch.client.engine import RetryConfig

logger = init_logger(__name__)
_T = TypeVar("_T")
POLL_INTERVAL_SECONDS = 2.0
CANCEL_TIMEOUT_SECONDS = 5.0


class VideoPersistenceError(InferenceError):
    """Keep the execution handle when the caller defers a storage failure."""


@dataclass
class VideoExecutionContext:
    deadline: float
    load: Callable[[], Awaitable[dict[str, Any]]]
    save: Callable[[dict[str, Any]], Awaitable[None]]
    persist: Callable[[BinaryIO, str], Awaitable[str]]
    cancelled: Callable[[], Awaitable[bool]]
    retry: RetryConfig


async def _storage_call(operation: Awaitable[_T]) -> _T:
    try:
        return await operation
    except Exception as exc:
        raise VideoPersistenceError(
            InferenceErrorCode.RESPONSE_ERROR,
            f"Video persistence failed: {exc}",
            retryable=False,
        ) from exc


class VideoExecutor:
    def __init__(
        self,
        client: HTTPXClientWrapper | httpx.AsyncClient,
        context: VideoExecutionContext,
        timeout: httpx.Timeout,
    ) -> None:
        self.client = client
        self.context = context
        self.timeout = timeout
        self.handle: dict[str, Any] = {}

    async def send(self, destination: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.handle = await _storage_call(
            asyncio.wait_for(self.context.load(), timeout=CANCEL_TIMEOUT_SECONDS)
        )
        self.context.deadline = min(
            self.context.deadline,
            self.handle.get("deadline", self.context.deadline),
        )
        # A result-file retry must survive backend teardown and job retention.
        if "response" in self.handle:
            return self.handle["response"]

        execution = asyncio.create_task(self._execute(destination, payload))
        cancellation = asyncio.create_task(self._wait_cancelled())
        try:
            done, _ = await asyncio.wait(
                {execution, cancellation},
                timeout=max(0.0, self.context.deadline - time.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution in done:
                return execution.result()
            if cancellation in done:
                cancellation.result()
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            await self._delete()
            raise InferenceError(
                InferenceErrorCode.RESPONSE_ERROR,
                "Video execution cancelled"
                if cancellation in done
                else "Video execution deadline exceeded",
                retryable=False,
            )
        finally:
            # Process interruption preserves the handle; only an explicit stop
            # or deadline above deletes the backend job.
            execution.cancel()
            cancellation.cancel()
            await asyncio.gather(execution, cancellation, return_exceptions=True)

    async def _wait_cancelled(self) -> None:
        while not await self.context.cancelled():
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _execute(
        self, destination: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if time.time() >= self.context.deadline or await self.context.cancelled():
            await self._delete()
            raise InferenceError(
                InferenceErrorCode.RESPONSE_ERROR,
                "Video execution cancelled or deadline exceeded",
                retryable=False,
            )
        if not self.handle.get("video_id"):
            if self.handle.get("submitting"):
                raise InferenceError(
                    InferenceErrorCode.RESPONSE_ERROR,
                    "Video creation outcome is unknown; refusing to replay POST",
                    retryable=False,
                )
            self.handle = {
                "destination": destination,
                "deadline": self.context.deadline,
                "submitting": True,
            }
            await _storage_call(self.context.save(self.handle))
            # filename=None produces text fields in multipart/form-data, even
            # without binary attachments. JSON values preserve Omni extensions.
            fields = {
                key: (None, value if isinstance(value, str) else json.dumps(value))
                for key, value in payload.items()
                if value is not None
            }
            try:
                response = await self.client.post(
                    self._url(), files=fields, timeout=self._request_timeout()
                )
                body = self._json(response)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                self.handle["submitting"] = False
                await _storage_call(self.context.save(self.handle))
                raise InferenceError(
                    InferenceErrorCode.TRANSPORT_ERROR, str(exc), retryable=True
                ) from exc
            except httpx.TransportError as exc:
                raise InferenceError(
                    InferenceErrorCode.TRANSPORT_ERROR,
                    f"Video creation outcome is unknown: {exc!r}; POST not replayed",
                    retryable=False,
                ) from exc
            except InferenceError as exc:
                # A gateway 5xx can follow backend acceptance/registration.
                if exc.status_code == 429:
                    self.handle["submitting"] = False
                    await _storage_call(self.context.save(self.handle))
                    exc.retryable = True
                raise
            video_id = body.get("id")
            if not isinstance(video_id, str) or not video_id:
                raise InferenceError(
                    InferenceErrorCode.RESPONSE_ERROR,
                    "Video create response has no ID; POST not replayed",
                    retryable=False,
                )
            self.handle["video_id"] = video_id
            self.handle.pop("submitting", None)
            await _storage_call(self.context.save(self.handle))

        while True:
            body = await self._retry(self._status)
            status = body.get("status")
            if status == "completed":
                output_file_id = await self._retry(self._download)
                body["output_file_id"] = output_file_id
                self.handle["output_file_id"] = output_file_id
                self.handle["response"] = body
                await _storage_call(self.context.save(self.handle))
                return body
            if status not in {"queued", "in_progress"}:
                raise InferenceError(
                    InferenceErrorCode.RESPONSE_ERROR,
                    f"Video {self.handle['video_id']} ended with status {status}: "
                    f"{body.get('error')}",
                    response_body=body,
                    retryable=False,
                )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    def _url(self, suffix: str = "") -> str:
        url = urljoin(self.handle["destination"], "/v1/videos")
        if self.handle.get("video_id"):
            url += "/" + quote(self.handle["video_id"], safe="")
        return url + suffix

    def _request_timeout(self) -> httpx.Timeout:
        # Keep transport setup bounds, but give generation/content reads the
        # remaining row budget instead of the channel's default 30 seconds.
        timeout = httpx.Timeout(self.timeout)
        timeout.read = max(0.001, self.context.deadline - time.time())
        return timeout

    def _json(self, response: httpx.Response) -> dict[str, Any]:
        self._check_status(response)
        try:
            body = response.json()
            if isinstance(body, dict):
                return body
        except ValueError:
            pass
        raise InferenceError(
            InferenceErrorCode.RESPONSE_ERROR,
            "Invalid Video JSON response",
            retryable=False,
        )

    def _check_status(self, response: httpx.Response) -> None:
        if response.is_error:
            raise InferenceError(
                InferenceErrorCode.HTTP_ERROR,
                f"{self._url()}: HTTP {response.status_code}",
                status_code=response.status_code,
                response_body=_response_body(response),
                retryable=False,
            )

    async def _status(self) -> dict[str, Any]:
        return self._json(
            await self.client.get(self._url(), timeout=self._request_timeout())
        )

    async def _download(self) -> str:
        async with self.client.stream(
            "GET", self._url("/content"), timeout=self._request_timeout()
        ) as response:
            if response.is_error:
                await response.aread()
                self._check_status(response)
            with tempfile.TemporaryFile() as content:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    await asyncio.to_thread(content.write, chunk)
                content.seek(0)
                return await _storage_call(
                    self.context.persist(
                        content, response.headers.get("content-type", "video/mp4")
                    )
                )

    async def _retry(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        policy = self.context.retry
        for attempt in range(policy.max_retries + 1):
            try:
                return await operation()
            except (httpx.TransportError, InferenceError) as exc:
                transient = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, InferenceError)
                    and exc.status_code is not None
                    and _is_retryable_status(exc.status_code)
                )
                if not transient or attempt == policy.max_retries:
                    if isinstance(exc, InferenceError):
                        raise
                    raise InferenceError(
                        InferenceErrorCode.TRANSPORT_ERROR,
                        f"{self._url()}: {exc!r}",
                        retryable=False,
                    ) from exc
                await asyncio.sleep(
                    min(
                        policy.base_delay_seconds * (2**attempt),
                        policy.max_delay_seconds,
                    )
                )
        raise AssertionError("unreachable")

    async def _delete(self) -> None:
        if not self.handle.get("video_id"):
            return
        try:
            response = await asyncio.wait_for(
                self.client.delete(self._url(), timeout=CANCEL_TIMEOUT_SECONDS),
                timeout=CANCEL_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Could not delete stopped video %s: %r", self._url(), exc)
