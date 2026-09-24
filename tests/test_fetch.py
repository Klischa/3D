from urllib.error import URLError
from types import SimpleNamespace
import pytest
onnx = pytest.importorskip("onnx")  # тяжёлая зависимость: пропуск без onnx
from pose_training import fetch_base as fetch


def test_download_falls_back_to_pinned_github_cli(monkeypatch):
    def blocked(*args, **kwargs):
        raise URLError('network unavailable')
    monkeypatch.setattr(fetch, 'urlopen', blocked)
    monkeypatch.setattr(fetch.shutil, 'which', lambda _: '/usr/bin/gh')
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'model bytes')
    monkeypatch.setattr(fetch.subprocess, 'run', run)
    assert fetch.download_base() == b'model bytes'
    assert fetch.BASE_GIT_BLOB in calls[0][0][2]
    assert calls[0][0][:2] == ['gh', 'api']
    assert not calls[0][1].get('shell', False)


def test_failed_download_without_cli_has_clear_error(monkeypatch):
    def blocked(*args, **kwargs):
        raise URLError('network unavailable')
    monkeypatch.setattr(fetch, 'urlopen', blocked)
    monkeypatch.setattr(fetch.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='GitHub CLI'):
        fetch.download_base()


def test_existing_unverified_file_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path/'original.onnx'
    path.write_bytes(b'not the verified base')
    def unexpected_download():
        raise AssertionError('Existing mismatched file must not trigger a download')
    monkeypatch.setattr(fetch, 'download_base', unexpected_download)
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        fetch.fetch_base(path)
    assert path.read_bytes() == b'not the verified base'
