from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Any

from scipy.signal import resample_poly

from ..exceptions import UserError
from ..logger import logger
from ..tracing import Span, SpeechGroupSpanData, speech_group_span, speech_span
from ..tracing.util import time_iso
from .events import (
    VoiceStreamEvent,
    VoiceStreamEventAudio,
    VoiceStreamEventError,
    VoiceStreamEventLifecycle,
)
from .imports import np, npt
from .model import TTSModel, TTSModelSettings
from .pipeline_config import VoicePipelineConfig

_FLUSH_BUFFER_SIZE = 3


def pcm_to_ulaw(pcm: np.ndarray) -> np.ndarray:
    """
    int16 PCM 데이터를 표준 G.711 μ-law (8비트, uint8) 포맷으로 변환합니다.
    """
    BIAS = 0x84  # 132, μ-law 변환 시 bias 값
    CLIP = 32635  # 클리핑 레벨

    # 연산 중 오버플로우를 막기 위해 int32로 변환
    pcm = pcm.astype(np.int32)
    pcm = np.clip(pcm, -32768, 32767)

    # 부호 추출 (음수인 경우 0x80, 양수면 0)
    sign = np.where(pcm < 0, 0x80, 0)

    # 절대값 취한 후 bias 적용
    pcm = np.abs(pcm) + BIAS
    pcm = np.clip(pcm, 0, CLIP)

    # 각 샘플에 대해 exponent 계산
    with np.errstate(divide="ignore"):  # 로그0에 대한 경고 무시
        exponent = np.floor(np.log2(pcm)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)

    # mantissa 계산: (exponent+3) 비트 오른쪽 시프트한 후 하위 4비트 추출
    mantissa = (pcm >> (exponent + 3)) & 0x0F

    # 부호, exponent, mantissa 합친 후 1의 보수 처리
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF

    return ulaw_byte.astype(np.uint8)


def resample_pcm_to_8kHz(
    pcm: np.ndarray, orig_sr: int = 24000, target_sr: int = 8000
) -> np.ndarray:
    """
    입력 PCM (int16) 배열을 원본 샘플레이트(orig_sr)에서 타겟 샘플레이트(target_sr)로 리샘플링합니다.
    scipy.signal.resample_poly 함수를 사용하여 효율적인 필터링과 함께 리샘플링합니다.
    """
    # up=1, down=3인 경우 24000 → 8000
    resampled = resample_poly(pcm, up=1, down=orig_sr // target_sr)
    return resampled.astype(np.int16)


def _audio_to_base64(audio_data: list[bytes]) -> str:
    joined_audio_data = b"".join(audio_data)
    return base64.b64encode(joined_audio_data).decode("utf-8")


class StreamedAudioResult:
    """The output of a `VoicePipeline`. Streams events and audio data as they're generated."""

    def __init__(
        self,
        tts_model: TTSModel,
        tts_settings: TTSModelSettings,
        voice_pipeline_config: VoicePipelineConfig,
    ):
        """Create a new `StreamedAudioResult` instance.

        Args:
            tts_model: The TTS model to use.
            tts_settings: The TTS settings to use.
            voice_pipeline_config: The voice pipeline config to use.
        """
        self.tts_model = tts_model
        self.tts_settings = tts_settings
        self.total_output_text = ""
        self.instructions = tts_settings.instructions
        self.text_generation_task: asyncio.Task[Any] | None = None

        self._voice_pipeline_config = voice_pipeline_config
        self._text_buffer = ""
        self._turn_text_buffer = ""
        self._queue: asyncio.Queue[VoiceStreamEvent] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []
        self._ordered_tasks: list[
            asyncio.Queue[VoiceStreamEvent | None]
        ] = []  # New: list to hold local queues for each text segment

        # Task to dispatch audio chunks in order – 인스턴스 생성 직후 바로 시작
        self._dispatcher_task: asyncio.Task[Any] = asyncio.create_task(self._dispatch_audio())


        self._done_processing = False
        self._buffer_size = tts_settings.buffer_size
        self._flush_buffer_size = tts_settings.flush_buffer_size
        self._started_processing_turn = False
        self._first_byte_received = False
        self._generation_start_time: str | None = None
        self._completed_session = False
        self._stored_exception: BaseException | None = None
        self._tracing_span: Span[SpeechGroupSpanData] | None = None

    async def _start_turn(self):
        if self._started_processing_turn:
            return

        self._tracing_span = speech_group_span()
        self._tracing_span.start()
        self._started_processing_turn = True
        self._first_byte_received = False
        self._generation_start_time = time_iso()
        await self._queue.put(VoiceStreamEventLifecycle(event="turn_started"))

    def _set_task(self, task: asyncio.Task[Any]):
        self.text_generation_task = task

    async def _add_error(self, error: Exception):
        await self._queue.put(VoiceStreamEventError(error))

    def _transform_audio_buffer(
        self, buffer: list[bytes], output_dtype: npt.DTypeLike
    ) -> npt.NDArray[np.int16 | np.float32]:
        np_array = np.frombuffer(b"".join(buffer), dtype=np.int16)
        np_array = resample_pcm_to_8kHz(np_array, orig_sr=24000, target_sr=8000)
        return pcm_to_ulaw(np_array)

    async def _stream_audio(
        self,
        text: str,
        local_queue: asyncio.Queue[VoiceStreamEvent | None],
        finish_turn: bool = False,
    ):
        with speech_span(
            model=self.tts_model.model_name,
            input=text
            if self._voice_pipeline_config.trace_include_sensitive_data
            else "",
            model_config={
                "voice": self.tts_settings.voice,
                "instructions": self.instructions,
                "speed": self.tts_settings.speed,
            },
            output_format="pcm",
            parent=self._tracing_span,
        ) as tts_span:
            try:
                first_byte_received = False
                buffer: list[bytes] = []
                full_audio_data: list[bytes] = []

                async for chunk in self.tts_model.run(text, self.tts_settings):
                    if not first_byte_received:
                        first_byte_received = True
                        tts_span.span_data.first_content_at = time_iso()

                    if chunk:
                        buffer.append(chunk)
                        full_audio_data.append(chunk)
                        if len(buffer) >= self._flush_buffer_size:
                            audio_np = self._transform_audio_buffer(
                                buffer, self.tts_settings.dtype
                            )
                            if self.tts_settings.transform_data:
                                audio_np = self.tts_settings.transform_data(audio_np)
                            await local_queue.put(
                                VoiceStreamEventAudio(data=audio_np)
                            )  # Use local queue
                            buffer = []
                if buffer:
                    audio_np = self._transform_audio_buffer(
                        buffer, self.tts_settings.dtype
                    )
                    if self.tts_settings.transform_data:
                        audio_np = self.tts_settings.transform_data(audio_np)
                    await local_queue.put(
                        VoiceStreamEventAudio(data=audio_np)
                    )  # Use local queue

                if self._voice_pipeline_config.trace_include_sensitive_audio_data:
                    tts_span.span_data.output = _audio_to_base64(full_audio_data)
                else:
                    tts_span.span_data.output = ""

                if finish_turn:
                    await local_queue.put(VoiceStreamEventLifecycle(event="turn_ended"))
                else:
                    await local_queue.put(None)  # Signal completion for this segment
            except Exception as e:
                tts_span.set_error(
                    {
                        "message": str(e),
                        "data": {
                            "text": text
                            if self._voice_pipeline_config.trace_include_sensitive_data
                            else "",
                        },
                    }
                )
                logger.error(f"Error streaming audio: {e}")

                # Signal completion for whole session because of error
                await local_queue.put(VoiceStreamEventLifecycle(event="session_ended"))
                raise e

    async def _add_text(self, text: str):
        await self._start_turn()

        self._text_buffer += text
        self.total_output_text += text
        self._turn_text_buffer += text

        combined_sentences, self._text_buffer = self.tts_settings.text_splitter(
            self._text_buffer
        )

        if len(combined_sentences) >= 1:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(local_queue)
            self._tasks.append(
                asyncio.create_task(self._stream_audio(combined_sentences, local_queue))
            )

    async def _turn_intercepted(self):
        self._text_buffer = ""
        self._done_processing = False

    async def _turn_done(self):
        if self._text_buffer:
            local_queue: asyncio.Queue[VoiceStreamEvent | None] = asyncio.Queue()
            self._ordered_tasks.append(
                local_queue
            )  # Append the local queue for the final segment
            self._tasks.append(
                asyncio.create_task(
                    self._stream_audio(self._text_buffer, local_queue, finish_turn=True)
                )
            )
            self._text_buffer = ""
        self._done_processing = True

    def _finish_turn(self):
        if self._tracing_span:
            if self._voice_pipeline_config.trace_include_sensitive_data:
                self._tracing_span.span_data.input = self._turn_text_buffer
            else:
                self._tracing_span.span_data.input = ""

            self._tracing_span.finish()
            self._tracing_span = None
        self._turn_text_buffer = ""
        self._started_processing_turn = False

    async def _done(self):
        self._completed_session = True
        await self._wait_for_completion()

    # async def _dispatch_audio(self):
    #     # Dispatch audio chunks from each segment in the order they were added
    #     while True:
    #         if len(self._ordered_tasks) == 0:
    #             if self._completed_session:
    #                 break
    #             await asyncio.sleep(0)
    #             continue
    #         local_queue = self._ordered_tasks.pop(0)
    #         while True:
    #             chunk = await local_queue.get()
    #             if chunk is None:
    #                 break
    #             await self._queue.put(chunk)
    #             if isinstance(chunk, VoiceStreamEventLifecycle):
    #                 local_queue.task_done()
    #                 if chunk.event == "turn_ended":
    #                     self._finish_turn()
    #                     break
    #     await self._queue.put(VoiceStreamEventLifecycle(event="session_ended"))

    async def _dispatch_audio(self):
        # ordered_queues가 비고, turn_done()으로 _done_processing=True가 세팅될 때까지 반복
        while True:
            # 전체 처리 완료 조건: 모든 텍스트 청크가 들어오고, ordered_queues가 비었을 때
            if self._completed_session and not self._ordered_tasks:
                break

            # 새 큐가 없으면 잠시 대기
            if not self._ordered_tasks:
                await asyncio.sleep(0.01)
                continue
            
            logger.warning("Dispatching audio from ordered tasks..."                           )
            # 새로 들어온 segment 큐를 꺼내서 그 안의 이벤트를 하나씩 처리
            queue = self._ordered_tasks.pop(0)
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                await self._queue.put(chunk)
                if isinstance(chunk, VoiceStreamEventLifecycle):
                    queue.task_done()
                    if chunk.event == "turn_ended":
                        self._finish_turn()
                        break

        # 모든 세그먼트가 처리되면 세션 종료 이벤트
        await self._queue.put(VoiceStreamEventLifecycle(event="session_ended"))

    async def _wait_for_completion(self):
        tasks: list[asyncio.Task[Any]] = self._tasks
        if self._dispatcher_task is not None:
            tasks.append(self._dispatcher_task)
        await asyncio.gather(*tasks)

    def _cleanup_tasks(self):
        self._finish_turn()

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()

        if self.text_generation_task and not self.text_generation_task.done():
            self.text_generation_task.cancel()

    def _check_errors(self):
        for task in self._tasks:
            if task.done():
                if task.exception():
                    self._stored_exception = task.exception()
                    break

    async def stream(self) -> AsyncIterator[VoiceStreamEvent]:
        """Stream the events and audio data as they're generated."""
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break
            if isinstance(event, VoiceStreamEventError):
                self._stored_exception = event.error
                logger.error(f"Error processing output: {event.error}")
                break
            if event is None:
                break
            yield event
            if (
                event.type == "voice_stream_event_lifecycle"
                and event.event == "session_ended"
            ):
                break

        self._check_errors()
        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception
