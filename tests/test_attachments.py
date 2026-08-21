import pytest

from fasteval.config import MAX_ATTACHMENT_BYTES, RunConfig
from fasteval.exceptions import ConfigError
from fasteval.providers import build_messages


def write_file(tmp_path, name: str, payload: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


def test_plain_prompt_is_plain_message():
    messages = build_messages("hello", None)
    assert messages == [{"role": "user", "content": "hello"}]


def test_image_becomes_image_url_part(tmp_path):
    # 1x1 transparent PNG
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
    )
    image_path = write_file(tmp_path, "pixel.png", png)
    content = build_messages("describe", [image_path])[0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_pdf_becomes_file_part(tmp_path):
    pdf_path = write_file(tmp_path, "invoice.pdf", b"%PDF-1.4 fake")
    content = build_messages("extract", [pdf_path])[0]["content"]
    assert content[1]["type"] == "file"
    assert content[1]["file"]["filename"] == "invoice.pdf"
    assert content[1]["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_text_file_is_inlined(tmp_path):
    text_path = write_file(tmp_path, "notes.md", b"# Notes\ncontent")
    content = build_messages("summarize", [text_path])[0]["content"]
    assert content[1]["type"] == "text"
    assert "notes.md" in content[1]["text"]
    assert "content" in content[1]["text"]


def test_oversized_attachment_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("fasteval.providers.MAX_ATTACHMENT_BYTES", 8)
    big_path = write_file(tmp_path, "pixel.png", b"1234567890")
    with pytest.raises(ConfigError, match="exceeds"):
        build_messages("x", [big_path])
    assert MAX_ATTACHMENT_BYTES > 0


def test_binary_attachment_rejected(tmp_path):
    bin_path = write_file(tmp_path, "blob.bin", bytes(range(256)))
    with pytest.raises(ConfigError, match="Unsupported attachment type"):
        build_messages("x", [bin_path])


def test_run_config_rejects_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="file not found"):
        RunConfig(prompt="x", file=str(tmp_path / "missing.txt"))
