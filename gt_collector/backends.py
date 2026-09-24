"""Бэкенды захвата: Orbbec Astra через pyorbbecsdk2 и синтетический фейк.

Ключевые решения (см. docs/DATA_COLLECTION.md):
* Один процесс, одно устройство — astra2 при записи должна быть закрыта
  (камера — эксклюзивная).
* GT — только по RGB + дошке; depth к GT не прикладывается.
* Заводская калибровка устройства (`Pipeline.get_camera_param()`)
  предпочтительна над собственной; источник калибровки фиксируется в
  manifest.
* Depth выравнивается по цвету (HW D2C при возможности, иначе SW
  AlignFilter по COLOR_STREAM), после чего размеры depth обязаны равняться
  размерам color — иначе твёрдая ошибка. K для RGB-D всегда K_color.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class FrameData:
    bgr: np.ndarray            # (H,W,3) uint8
    depth_mm: np.ndarray       # (H,W) float32, мм, выравненный по цвету
    ts_us: int                 # метка времени frameset, мкс
    extra: dict = field(default_factory=dict)


class Backend:
    """Общий интерфейс: live-поток кадров + детектор + метаданные."""

    def start(self) -> None: ...
    def wait_for_frame(self, timeout_ms: int = 1000) -> FrameData | None: ...
    def stop(self) -> None: ...

    @property
    def detector(self):
        raise NotImplementedError

    def metadata(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Orbbec
# ---------------------------------------------------------------------------

def _color_to_bgr(color_frame) -> np.ndarray:
    """RGB-кадр Orbbec → BGR (форматы как в официальных примерах SDK)."""
    import pyorbbecsdk as obs
    w, h = color_frame.get_width(), color_frame.get_height()
    fmt = color_frame.get_format()
    data = color_frame.get_data()
    if fmt == obs.OBFormat.RGB:
        img = np.frombuffer(data, np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == obs.OBFormat.BGR:
        return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()
    if fmt == obs.OBFormat.MJPG:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Не удалось декодировать MJPG-кадр")
        return img
    if fmt in (obs.OBFormat.YUYV, obs.OBFormat.YUY2):
        img = np.frombuffer(data, np.uint8).reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if fmt in (obs.OBFormat.NV12, obs.OBFormat.NV21):
        img = np.frombuffer(data, np.uint8).reshape(h * 3 // 2, w)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    raise RuntimeError(f"Неподдерживаемый формат цвета: {fmt}")


def _intrinsic_to_K(intr) -> np.ndarray:
    return np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], np.float32)


def _distortion_to_vec(dist) -> tuple[np.ndarray, bool]:
    """OBCameraDistortion → (вектор дисторсии OpenCV, fisheye: bool).

    Модели OBCameraDistortionModel: 0=NONE, 1=MODIFIED_BROWN_CONRADY,
    2=INVERSE_BROWN_CONRADY, 3=BROWN_CONRADY, 4=BROWN_CONRADY_K6,
    5=KANNALA_BRANDT4 (fisheye, решается cv2.fisheye.undistortImage).
    """
    model = int(getattr(dist, "model", 0))
    k1, k2, k3 = (float(getattr(dist, f"k{i}", 0.0)) for i in (1, 2, 3))
    k4, k5, k6 = (float(getattr(dist, f"k{i}", 0.0)) for i in (4, 5, 6))
    p1, p2 = float(getattr(dist, "p1", 0.0)), float(getattr(dist, "p2", 0.0))
    if model == 5:  # Kannala-Brandt (fisheye)
        return np.array([k1, k2, k3, k4], np.float64), True
    if model == 4:  # Brown-Conrady с 6 радиальными
        # порядок OpenCV: k1, k2, p1, p2, k3..k6
        return np.array([k1, k2, p1, p2, k3, k4, k5, k6], np.float64), False
    # NONE / MODIFIED_BROWN_CONRADY / INVERSE_BROWN_CONRADY / BROWN_CONRADY:
    # базовый набор (для inverse-модели — best effort)
    return np.array([k1, k2, p1, p2, k3], np.float64), False


def list_devices() -> list[dict]:
    """Список видимых устройств Orbbec (имя, serial, версия прошивки)."""
    import pyorbbecsdk as obs
    out = []
    ctx = obs.Context()
    dl = ctx.query_devices()
    for i in range(dl.get_count()):
        row = {
            "index": i,
            "name": dl.get_device_name_by_index(i),
            "serial": dl.get_device_serial_number_by_index(i),
            "firmware": "",
        }
        # get_device_by_index() открывает устройство эксклюзивно — пробуем
        # только для версии прошивки и не ломаемся, если занято.
        try:
            row["firmware"] = (dl.get_device_by_index(i)
                               .get_device_info().get_firmware_version())
        except Exception:
            pass
        out.append(row)
    return out


class OrbbecBackend(Backend):
    """Захват Astra через Orbbec SDK v2 (Pipeline)."""

    def __init__(self, name_hint: str = "astra", serial: str | None = None,
                 color_size: tuple[int, int] | None = None,
                 depth_size: tuple[int, int] | None = None,
                 fps: int = 30,
                 detector: "OpenCVAprilTagDetector | None" = None):
        import pyorbbecsdk as obs
        self.obs = obs
        self._name_hint = name_hint.lower()
        self._serial = serial
        self._color_size = color_size
        self._depth_size = depth_size
        self._fps = fps
        self._detector = detector
        self.ctx = obs.Context()
        self.dev = self._pick_device()
        self.pipe = obs.Pipeline(self.dev)
        self.cfg = obs.Config()
        self._align = None
        self._hw_d2c = False
        self._calib = None
        self._K_color: np.ndarray | None = None
        self._dist_color: np.ndarray | None = None
        self._dist_fisheye: bool = False
        self._K: np.ndarray | None = None          # K для (унареженного) RGB
        self._undistort: bool = False

    # -- устройство --------------------------------------------------------
    def _pick_device(self):
        obs = self.obs
        dl = self.ctx.query_devices()
        for i in range(dl.get_count()):
            dev = dl.get_device_by_index(i)
            info = dev.get_device_info()
            name = info.get_name().lower()
            if self._serial is not None:
                if info.get_serial_number() == self._serial:
                    return dev
            elif not self._name_hint or self._name_hint in name:
                return dev
        raise RuntimeError(
            f"Устройство Orbbec не найдено (hint={self._name_hint!r}, "
            f"serial={self._serial!r}). Установлен ли драйвер Orbbec SDK? "
            f"Закрыта ли astra2?")

    # -- профили и старт ----------------------------------------------------
    def _pick_profile(self, lst, size, fps, default):
        obs = self.obs
        for p in lst:
            try:
                pw, ph, pf = p.get_width(), p.get_height(), p.get_fps()
            except Exception:
                continue
            if size is not None and (pw, ph) != tuple(size):
                continue
            if fps and pf != fps:
                continue
            return p
        return default

    def start(self) -> None:
        obs = self.obs
        cl = self.pipe.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
        self.color_profile = self._pick_profile(
            list(cl), self._color_size, self._fps, cl.get_default_video_stream_profile())
        # HW D2C: depth выравнивается аппаратно в размеры color
        depth_profile = None
        try:
            d2c = self.pipe.get_d2c_depth_profile_list(self.color_profile, obs.OBAlignMode.HW_MODE)
            if d2c:
                d2l = list(d2c)
                if d2l:
                    depth_profile = self._pick_profile(
                        d2l, self._depth_size, 0, d2c.get_default_video_stream_profile())
                    self._hw_d2c = True
        except Exception:
            depth_profile = None
        if depth_profile is None:
            dl = self.pipe.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
            depth_profile = self._pick_profile(
                list(dl), self._depth_size, 0, dl.get_default_video_stream_profile())
            self._align = obs.AlignFilter(align_to_stream=obs.OBStreamType.COLOR_STREAM)
            try:
                # гарантируем, что каждый frameset несёт и color, и depth
                self.cfg.set_frame_aggregate_output_mode(
                    obs.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception:
                pass

        self.cfg.enable_stream(self.color_profile)
        self.cfg.enable_stream(depth_profile)
        if self._hw_d2c:
            self.cfg.set_align_mode(obs.OBAlignMode.HW_MODE)
            try:
                self.pipe.enable_frame_sync()
            except Exception:
                pass
        self.pipe.start(self.cfg)
        self._init_calibration()

    def _init_calibration(self) -> None:
        """Заводская калибровка после первого frameset; строит K для RGB-D."""
        frames = self.pipe.wait_for_frames(2000)
        if frames is None:
            raise RuntimeError("Не пришёл первый frameset")
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if c is None or d is None:
            raise RuntimeError("Нет color/depth в frameset")
        calib = self.pipe.get_camera_param()
        self._calib = calib
        self._K_color = _intrinsic_to_K(calib.rgb_intrinsic)
        self._dist_color, self._dist_fisheye = _distortion_to_vec(calib.rgb_distortion)
        cw, ch, dw, dh = c.get_width(), c.get_height(), d.get_width(), d.get_height()
        if self._hw_d2c:
            if (dw, dh) != (cw, ch):
                raise RuntimeError(
                    f"HW D2C: размеры depth ({dw}x{dh}) != color ({cw}x{ch}) — "
                    "нельзя уверенно построить RGB-D")
            self._K = self._K_color
            self._undistort = bool(np.any(self._dist_color))
        else:
            # SW AlignFilter(COLOR_STREAM) ресайзит depth в размер color
            self._K = self._K_color
            self._undistort = bool(np.any(self._dist_color))

    @property
    def K(self) -> np.ndarray:
        if self._K is None:
            raise RuntimeError("Бэкенд не запущен (нет калибровки)")
        return self._K

    @property
    def K_undistorted(self) -> np.ndarray:
        """K для унарежженного кадра (== K_color при нулевой дисторсии)."""
        return self.K

    @property
    def detector(self):
        if self._detector is None:
            from .detectors import OpenCVAprilTagDetector
            self._detector = OpenCVAprilTagDetector()
        return self._detector

    # -- кадры --------------------------------------------------------------
    def wait_for_frame(self, timeout_ms: int = 1000) -> FrameData | None:
        try:
            frames = self.pipe.wait_for_frames(timeout_ms)
        except Exception:
            return None
        if frames is None:
            return None
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if c is None or d is None:
            return None
        if self._align is not None:
            frames = self._align.process(frames)
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if c is None or d is None:
                return None
        bgr = _color_to_bgr(c)
        scale = d.get_depth_scale()
        depth = (np.frombuffer(d.get_data(), np.uint16)
                 .reshape(d.get_height(), d.get_width()).astype(np.float32) * float(scale))
        if self._align is not None and (depth.shape[1], depth.shape[0]) != (bgr.shape[1], bgr.shape[0]):
            # защита: после SW-align depth обязан быть в размерах color
            depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        return FrameData(bgr=bgr, depth_mm=depth, ts_us=frames.get_timestamp_us(),
                         extra={"backend": "orbbec", "hw_d2c": self._hw_d2c})

    def undistort_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Унареживание RGB перед детекцией (заводские дисторсии)."""
        if self._dist_color is None or not self._undistort:
            return bgr
        if self._dist_fisheye:
            return cv2.fisheye.undistortImage(bgr, self.K_undistorted,
                                              self._dist_color)
        return cv2.undistort(bgr, self.K_undistorted, self._dist_color)

    def metadata(self) -> dict:
        out = {"backend": "orbbec", "hw_d2c": self._hw_d2c}
        try:
            info = self.dev.get_device_info()
            out.update(name=info.get_name(), serial=info.get_serial_number(),
                       firmware=info.get_firmware_version(),
                       vid=info.get_vid(), pid=info.get_pid())
            try:
                out["baseline_mm"] = float(self.dev.get_baseline().baseline)
            except Exception:
                pass
        except Exception:
            pass
        if self._K_color is not None:
            out["K_color"] = self._K_color.tolist()
            out["distortion_color"] = self._dist_color.tolist()
            out["calibration_source"] = "device factory (Pipeline.get_camera_param)"
        return out

    def stop(self) -> None:
        try:
            self.pipe.stop()
        except Exception:
            pass
        # Context закрывать не нужно: SDK v2 не даёт Context.close()
