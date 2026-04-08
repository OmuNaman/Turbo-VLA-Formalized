"""Configuration dataclasses for the recording pipeline."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotServerConfig:
    """Config for the robot-side HTTP server."""
    host: str = "0.0.0.0"
    port: int = 8080
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    watchdog_timeout_s: float = 0.5
    max_duty: float = 80.0
    jpeg_quality: int = 70


@dataclass
class RecordingConfig:
    """Config for the laptop-side recording client."""
    # Robot connection
    robot_ip: str = "192.168.149.1"
    robot_port: int = 8080

    # Dataset
    dataset_name: str = "turbopi_nav"
    repo_id: str = "<HF_DATASET_REPO>"
    robot_type: str = "turbopi"

    # Recording parameters
    fps: int = 10
    episode_time_s: float = 30.0
    num_episodes: int = 50

    # Teleop
    teleop_speed: float = 50.0
    max_duty: float = 80.0

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    session_name: str | None = None

    # Video encoding
    vcodec: str = "h264"
    video_fps: int = 10
    jpeg_quality: int = 70

    # Normalization range for state/action
    # Duty cycle range is [-max_duty, max_duty], normalized to [-1, 1]
    duty_range: float = 80.0

    @property
    def robot_url(self) -> str:
        return f"http://{self.robot_ip}:{self.robot_port}"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_name

    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def episodes_dir(self) -> Path:
        return self.dataset_dir / "episodes"

    @property
    def lerobot_dir(self) -> Path:
        return self.dataset_dir / "lerobot"


@dataclass
class ExportConfig:
    """Config for converting episodes to LeRobot v3.0 format."""
    episodes_dir: Path = field(default_factory=lambda: Path("data/turbopi_nav/episodes"))
    output_dir: Path = field(default_factory=lambda: Path("data/turbopi_nav/lerobot"))
    repo_id: str = "<HF_DATASET_REPO>"
    robot_type: str = "turbopi"
    fps: int = 10
    image_key: str = "observation.images.front"
    state_source: str = "shifted_action"
    vcodec: str = "h264"
    overwrite: bool = False
    push_to_hub: bool = False
