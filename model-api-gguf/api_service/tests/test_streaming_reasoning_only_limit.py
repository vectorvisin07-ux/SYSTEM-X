from __future__ import annotations

import unittest

from system_x_gguf_api.sse import ValidatedPrivateFrame
from system_x_gguf_api.streaming_inference import (
    OpenAIPrivateStreamNormalizer,
    StreamNormalizationError,
)
from system_x_gguf_api.stream_types import (
    ActiveStreamState,
    CanonicalStreamEventType,
)


def frame(value: dict | None = None, *, done: bool = False) -> ValidatedPrivateFrame:
    return ValidatedPrivateFrame(
        event=None,
        id=None,
        value=value,
        done=done,
        heartbeat=False,
        comments=(),
    )


def normalizer() -> OpenAIPrivateStreamNormalizer:
    state = ActiveStreamState(
        request_id="sx_req_0123456789abcdef0123456789abcdef",
        endpoint="/v1/chat/completions",
        model="sx-gguf-test",
    )
    state.begin_backend()
    state.attach_upstream()
    state.start("chat")
    return OpenAIPrivateStreamNormalizer(state, "chat", [], None)


class ReasoningOnlyLimitTests(unittest.TestCase):
    def test_reasoning_only_length_is_truthful_incomplete(self) -> None:
        value = normalizer()
        reasoning = value.accept(
            frame(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "bounded thought"},
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
            )
        )
        self.assertEqual(
            [event.type for event in reasoning],
            [CanonicalStreamEventType.REASONING_DELTA],
        )
        value.accept(
            frame(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": None,
                },
            )
        )
        value.accept(
            frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 64,
                        "total_tokens": 74,
                    },
                },
            )
        )
        terminal = value.accept(frame(done=True))
        self.assertEqual(
            [event.type for event in terminal],
            [
                CanonicalStreamEventType.USAGE,
                CanonicalStreamEventType.INCOMPLETE,
            ],
        )
        self.assertEqual(
            terminal[-1].payload,
            {
                "status": "incomplete",
                "finish_reason": "output_limit",
            },
        )

    def test_reasoning_only_stop_remains_invalid(self) -> None:
        value = normalizer()
        value.accept(
            frame(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "thought"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 1,
                        "total_tokens": 11,
                    },
                }
            )
        )
        with self.assertRaisesRegex(
            StreamNormalizationError, "reasoning_only_output"
        ):
            value.accept(frame(done=True))


if __name__ == "__main__":
    unittest.main()
