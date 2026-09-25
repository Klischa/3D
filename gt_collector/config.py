"""Конфигурации дошки с маркерами и сессии. Всё в метрах/секундах."""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class BoardSpec:
    """Жёсткая дошка с AprilTag: cols×rows маркеров, равномерная сетка.

    Система координат дошки (мировая для сессии): начало — в центре дошки,
    X вправо, Y вниз, Z из лицевой (печатной) стороны.
    """
    cols: int = 4
    rows: int = 3
    tag_mm: float = 50.0
    gap_mm: float = 10.0
    margin_mm: float = 15.0

    @classmethod
    def presets(cls) -> dict:
        return {
            # 12 крупных маркеров на A4 (297×210) — базовый, удобный вариант.
            "a4": BoardSpec(4, 3, 50.0, 10.0, 15.0),
            # 20 маркеров на A3 (420×297) — для больших сцен.
            "a3": BoardSpec(5, 4, 50.0, 15.0, 15.0),
            # 25 маркеров на жёсткой дошке 300×300 мм.
            "square300": BoardSpec(5, 5, 40.0, 15.0, 20.0),
        }

    @classmethod
    def from_preset(cls, name: str) -> "BoardSpec":
        try:
            return cls.presets()[name]
        except KeyError:
            raise ValueError(f"Неизвестный пресет дошки: {name!r}. Доступны: {sorted(cls.presets())}")

    def __post_init__(self):
        if self.cols < 2 or self.rows < 2 or self.tag_mm <= 0 or self.gap_mm < 0 or self.margin_mm < 0:
            raise ValueError("Некорректные размеры дошки")
        if self.cols * self.rows > 25:
            raise ValueError("В словаре DICT_APRILTAG_25H9 только 25 маркеров")

    @property
    def tag_count(self) -> int:
        return self.cols * self.rows

    @property
    def pitch_m(self) -> float:
        return (self.tag_mm + self.gap_mm) / 1000.0

    @property
    def width_m(self) -> float:
        return self.margin_mm * 2 / 1000.0 + self.cols * self.tag_mm / 1000.0 + (self.cols - 1) * self.gap_mm / 1000.0

    @property
    def height_m(self) -> float:
        return self.margin_mm * 2 / 1000.0 + self.rows * self.tag_mm / 1000.0 + (self.rows - 1) * self.gap_mm / 1000.0

    @property
    def tag_ids(self) -> list[int]:
        """ID маркеров в порядке сетки (строка за строкой, с левого верхнего)."""
        return list(range(self.tag_count))

    def tag_centers_m(self) -> "np.ndarray":  # noqa: F821
        import numpy as np
        n = self.tag_count
        out = np.zeros((n, 3), dtype=np.float64)
        step = self.pitch_m
        for i in range(self.rows):
            for j in range(self.cols):
                k = i * self.cols + j
                out[k, 0] = (j - (self.cols - 1) / 2.0) * step
                out[k, 1] = (i - (self.rows - 1) / 2.0) * step
                out[k, 2] = 0.0
        return out

    def size(self) -> dict:
        return {"cols": self.cols, "rows": self.rows, "tag_mm": self.tag_mm,
                "gap_mm": self.gap_mm, "margin_mm": self.margin_mm,
                "width_m": self.width_m, "height_m": self.height_m}


@dataclass
class SessionConfig:
    """Одна запись: дошка + траектория оператора.

    Ограничения движения на кадр (30 fps): по умолчанию не более 0,2 м и 20° —
    это верхняя граница; рабочие данные лучше держать в ~10 см и ~10°.
    """
    board: BoardSpec = field(default_factory=lambda: BoardSpec.from_preset("a4"))
    duration_s: float = 60.0
    frames: int | None = None  # None = по duration_s
    warmup_frames: int = 20
    save_frames_every: int = 1  # писать .jpg/.cloud каждый N-й кадр (позы — каждый)

    min_tags: int = 5             # минимум видимых маркеров в кадре
    max_residual_mm: float = 20.0 # максимум среднего резидуала Umeyama
    min_points: int = 20_000      # минимум точек в облаке
    max_step_translation_m: float = 0.2
    max_step_rotation_deg: float = 20.0
    max_frame_gap_ms: float = 80.0  # разрыв по времени между кадрами

    depth_min_m: float = 0.3
    depth_max_m: float = 4.0

    scene: str = ""
    operator: str = ""
    calibration: str | None = None  # путь к calibration.json (K+дисторсии RGB)

    # Устройства
    device_name_hint: str = "astra"   # подстрока имени устройства (Orbbec SDK)
    device_serial: str | None = None
    fps: int = 30

    def validate(self):
        if self.duration_s <= 0 and not (self.frames and self.frames > 0):
            raise ValueError("Укажите duration_s или frames")
        if self.min_tags < 3:
            raise ValueError("min_tags должно быть >= 3")
        if not (0 < self.depth_min_m < self.depth_max_m):
            raise ValueError("depth_min_m должен быть меньше depth_max_m")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames >= 0")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionConfig":
        data = dict(data)
        data["board"] = BoardSpec(**data["board"])
        return cls(**data)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SessionConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
