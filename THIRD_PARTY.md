# 第三方来源与许可

1. Franka Emika Panda 的 MJCF 与网格来自 Google DeepMind [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，本地快照提交为 `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7`。源文件位于 `third_party/mujoco_menagerie/franka_emika_panda/`，其 `LICENSE` 保留原许可。机器人主体模型及原生双指夹爪均来自该项目；本项目在运行时扩展输送线、相机、零件、料箱、抓取中心标记与指尖接触参数，未改写原始 `panda.xml`。
2. MuJoCo 3.11.0、OpenCV、NumPy、SciPy、Matplotlib、ImageIO、imageio-ffmpeg 和 Pillow 由各自项目提供。运行依赖版本见 `requirements-full.txt`，用法见 `README.md`。
