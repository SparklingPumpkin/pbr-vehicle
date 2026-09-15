from __future__ import annotations

import copy
import json
import math
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import viser
import viser.transforms as vt

from .asset_io import load_scene, load_vehicle_asset, materialize_pth_scene, resolve_scene_context
from .auto_fit import run_vehicle_auto_fit
from .config_io import next_config_path, read_config, save_viewer_config, state_from_config
from .device import resolve_compute_device
from .environment_map import EnvironmentMap, cube_face_rotations, textured_environment_sphere
from .math3d import color_temperature_to_rgb, euler_to_wxyz, normalize, quaternion_to_rotation, sun_direction
from .projection import build_projection_masks
from .rendering import pack_gaussian_buffer, rgba_plane_glb
from .shading import shade_vehicle
from .sun_estimation import run_sun_estimator
from .types import LightingState, MaterialState, TransformState, VehicleAsset, VehicleState


DISPLAY_MODES = (
    "Relight Original", "Original SH", "Proxy Lit", "Albedo", "Normal",
    "Roughness", "Reflectance", "Metallic", "SH RGB",
)


def _scene_display_indices(total_splats: int, max_splats: int, seed: int) -> np.ndarray | slice:
    if max_splats <= 0 or total_splats <= max_splats:
        return slice(None)
    return np.sort(np.random.default_rng(seed).choice(total_splats, size=max_splats, replace=False))


def _ui_to_value(value: float, kind: str) -> float:
    value = float(value)
    if kind == "sun_intensity":
        value = float(np.clip(value, -1.0, 1.0))
        return 1.0 + value * (1.0 if value < 0.0 else 7.0)
    if kind == "brightness":
        value = float(np.clip(value, -1.0, 1.0))
        return 0.35 + value * (0.35 if value < 0.0 else 0.65)
    if kind == "temperature":
        value = float(np.clip(value, -0.5, 0.5))
        return 6500.0 + value * (9000.0 if value < 0.0 else 11000.0)
    if kind == "saturation":
        return float(np.clip(value, 0.0, 2.0))
    raise ValueError(kind)


def _value_to_ui(value: float, kind: str) -> float:
    value = float(value)
    if kind == "sun_intensity":
        result = (value - 1.0) / (1.0 if value < 1.0 else 7.0)
        return float(np.clip(result, -1.0, 1.0))
    if kind == "brightness":
        result = (value - 0.35) / (0.35 if value < 0.35 else 0.65)
        return float(np.clip(result, -1.0, 1.0))
    if kind == "temperature":
        result = (value - 6500.0) / (9000.0 if value < 6500.0 else 11000.0)
        return float(np.clip(result, -0.5, 0.5))
    if kind == "saturation":
        return float(np.clip(value, 0.0, 2.0))
    raise ValueError(kind)


class StandaloneViewer:
    def __init__(self, args):
        self.args = args
        self.compute_device = resolve_compute_device(getattr(args, "device", "auto"))
        print(f"PBR compute device: {self.compute_device.description}", flush=True)
        self.server = viser.ViserServer(port=args.port)
        self.scene_handle = None
        self.scene_path = ""
        self.scene_context = {"name": "empty_scene", "data_root": None, "config_path": None}
        self.scene_visible = True
        self.controllers: list[VehicleController] = []
        self.next_vehicle_index = 1
        self.handles: dict[str, Any] = {}
        self._environment_capture_lock = threading.RLock()
        self._environment_camera_lock = threading.Lock()
        self._environment_camera_timer = None
        self._environment_camera_generation = 0
        self._environment_generation = 0
        self._environment_client = None
        self._environment_client_ids: set[int] = set()
        self._sun_estimation_thread = None
        self._build_gui()
        self.server.on_client_connect(self._on_environment_client_connect)
        if args.scene:
            self.replace_scene(args.scene)
        if args.vehicle_asset_folder:
            self.add_vehicle(args.vehicle_asset_folder, args.config)

    def _notify(self, event, title: str, body: str) -> None:
        client = getattr(event, "client", None)
        if client is not None:
            client.add_notification(title=title, body=body, auto_close_seconds=5.0)

    def _build_gui(self) -> None:
        with self.server.gui.add_folder("Scene", expand_by_default=True):
            self.handles["scene_path"] = self.server.gui.add_text("Scene PLY/PTH path", initial_value=str(self.args.scene or ""))
            self.handles["scene_visible"] = self.server.gui.add_checkbox("Show scene", initial_value=True)
            load_button = self.server.gui.add_button("Load / replace scene")
            clear_button = self.server.gui.add_button("Clear scene")
            self.handles["sun_estimator_config"] = self.server.gui.add_text(
                "太阳估计配置",
                initial_value=str(getattr(self.args, "sun_estimator_config", "") or ""),
                hint=(
                    "JSON 配置可提供 pbr-vehicle-sun-scene 的 command 或模型权重路径；"
                    "估计结果写入临时目录并自动回填太阳角度。"
                ),
            )
            self.handles["estimate_sun"] = self.server.gui.add_button("太阳估计")
            self.handles["sun_inference_progress"] = self.server.gui.add_progress_bar(0.0)
            self.handles["sun_inference_status"] = self.server.gui.add_text("太阳估计状态", initial_value="等待启动", disabled=True)
            self.handles["sun_inference_stage"] = self.server.gui.add_text("当前阶段", initial_value="尚未执行", disabled=True)
            self.handles["sun_inference_result"] = self.server.gui.add_text("推理结果", initial_value="尚无结果", multiline=True, disabled=True)
        with self.server.gui.add_folder("Shared lighting", expand_by_default=True):
            self.handles["env_temperature"] = self.server.gui.add_slider("环境光色温", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
            self.handles["sun_intensity"] = self.server.gui.add_slider("太阳光强度", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
            self.handles["sun_azimuth"] = self.server.gui.add_slider("太阳方位角", min=-180.0, max=180.0, step=1.0, initial_value=45.0)
            self.handles["sun_elevation"] = self.server.gui.add_slider("太阳高度角", min=-10.0, max=89.0, step=1.0, initial_value=35.0)
            with self.server.gui.add_folder("次要参数", expand_by_default=False):
                self.handles["env_intensity"] = self.server.gui.add_slider("环境光强度", min=0.0, max=4.0, step=0.05, initial_value=1.0)
                self.handles["sun_r"] = self.server.gui.add_slider("Sun red", min=0.0, max=3.0, step=0.05, initial_value=1.0)
                self.handles["sun_g"] = self.server.gui.add_slider("Sun green", min=0.0, max=3.0, step=0.05, initial_value=1.0)
                self.handles["sun_b"] = self.server.gui.add_slider("Sun blue", min=0.0, max=3.0, step=0.05, initial_value=1.0)
            with self.server.gui.add_folder("高级", expand_by_default=False):
                self.handles["sun_enabled"] = self.server.gui.add_checkbox("Directional sun", initial_value=True)
                self.handles["visibility"] = self.server.gui.add_slider("Visibility scale", min=0.0, max=1.0, step=0.05, initial_value=1.0)
                reset_light = self.server.gui.add_button("Reset shared lighting")
        with self.server.gui.add_folder("Vehicles", expand_by_default=True):
            self.handles["asset_folder"] = self.server.gui.add_text("Vehicle asset folder", initial_value=str(self.args.vehicle_asset_folder or ""))
            self.handles["vehicle_config"] = self.server.gui.add_text("Initial config path (optional)", initial_value=str(self.args.config or ""))
            self.handles["auto_fit_config"] = self.server.gui.add_text(
                "Auto 识别配置",
                initial_value=str(getattr(self.args, "auto_fit_config", "") or ""),
                hint="可选 JSON；默认调用 pbr-vehicle-auto-fit。过程文件写入临时目录。",
            )
            add_button = self.server.gui.add_button("Add vehicle")

        load_button.on_click(lambda event: self._load_scene_event(event))
        clear_button.on_click(lambda _: self.clear_scene())
        self.handles["estimate_sun"].on_click(lambda event: self._estimate_sun_event(event))
        self.handles["scene_visible"].on_update(lambda _: self._sync_scene_visibility())
        add_button.on_click(lambda event: self._add_vehicle_event(event))
        for key in ("env_temperature", "env_intensity", "sun_enabled", "sun_intensity", "sun_r", "sun_g", "sun_b", "sun_azimuth", "sun_elevation", "visibility"):
            self.handles[key].on_update(lambda _, source=key: self._shared_lighting_changed(source))
        reset_light.on_click(lambda _: self.apply_shared_lighting(LightingState()))

    def _load_scene_event(self, event) -> None:
        try:
            self.replace_scene(self.handles["scene_path"].value)
            self._notify(event, "Scene loaded", self.scene_path)
        except Exception as exc:
            traceback.print_exc()
            self._notify(event, "Scene load failed", f"{type(exc).__name__}: {exc}")

    def _estimate_sun_event(self, event) -> None:
        """Start SSE without blocking Viser's GUI event thread."""
        if self._sun_estimation_thread is not None and self._sun_estimation_thread.is_alive():
            self._notify(event, "太阳估计进行中", "请等待当前估计完成")
            return
        self.handles["estimate_sun"].disabled = True
        self._set_sun_inference_progress(0.0, "运行中", "准备当前场景输入", "等待预测结果")
        self._notify(event, "太阳估计已启动", "模型正在运行，完成后会自动回填太阳角度")
        self._sun_estimation_thread = threading.Thread(
            target=self._run_sun_estimation,
            args=(event,),
            daemon=True,
        )
        self._sun_estimation_thread.start()

    def _set_sun_inference_progress(self, progress, status, stage, result=None) -> None:
        self.handles["sun_inference_progress"].value = 100.0 * float(np.clip(progress, 0.0, 1.0))
        self.handles["sun_inference_status"].value = str(status)
        self.handles["sun_inference_stage"].value = str(stage)
        if result is not None:
            self.handles["sun_inference_result"].value = str(result)

    def _run_sun_estimation(self, event) -> None:
        try:
            scene_path = str(self.handles["scene_path"].value).strip()
            if not scene_path and not self.scene_path:
                raise ValueError("请先加载或填写场景路径")
            result = run_sun_estimator(
                config_path=str(self.handles["sun_estimator_config"].value).strip() or None,
                scene=str(getattr(self.args, "sun_estimator_scene", "") or self.scene_context["name"]),
                data_root=str(getattr(self.args, "sun_estimator_data_root", "") or self.scene_context.get("data_root") or ""),
                scene_path=self.scene_path or scene_path,
                timeout=float(getattr(self.args, "sun_estimator_timeout", 3600.0)),
                progress_callback=lambda progress, stage: self._set_sun_inference_progress(progress, "运行中", stage),
            )
            world_direction = result.get("world_direction")
            if world_direction is not None:
                direction = normalize(np.asarray(world_direction, dtype=np.float32).reshape(1, 3))[0]
                estimated_azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
                estimated_elevation = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
            else:
                estimated_azimuth = float(result["azimuth_deg"])
                estimated_elevation = float(result["elevation_deg"])
            values = {
                "sun_enabled": True,
                "sun_azimuth": ((estimated_azimuth + 180.0) % 360.0) - 180.0,
                "sun_elevation": float(np.clip(estimated_elevation, -10.0, 89.0)),
            }
            payload = result.get("payload", {})
            weighted = payload.get("weighted_angle", {}) if isinstance(payload, dict) else {}
            if isinstance(weighted, dict) and "sun_intensity" in weighted:
                values["sun_intensity"] = _value_to_ui(weighted["sun_intensity"], "sun_intensity")
            self.apply_estimated_sun_angles(values["sun_azimuth"], values["sun_elevation"])
            if "sun_intensity" in values:
                self.handles["sun_intensity"].value = values["sun_intensity"]
                self._shared_lighting_changed("sun_estimate")
            published_round = result.get("published_round")
            round_text = f"；发布轮次 {published_round}" if published_round is not None else ""
            result_text = (
                f"太阳方位角：{values['sun_azimuth']:.1f}°\n"
                f"太阳高度角：{values['sun_elevation']:.1f}°{round_text}\n"
                f"结果目录：{result['output_dir']}"
            )
            self._set_sun_inference_progress(1.0, "成功", "角度已通过发布门并回填面板", result_text)
            self._notify(
                event,
                "太阳估计完成",
                f"方位角 {values['sun_azimuth']:.1f}°，高度角 {values['sun_elevation']:.1f}°；结果保留在 {result['output_dir']}",
            )
        except Exception as exc:
            traceback.print_exc()
            self._set_sun_inference_progress(1.0, "失败", "推理终止，面板参数未修改", f"{type(exc).__name__}: {exc}")
            self._notify(event, "太阳估计失败", f"{type(exc).__name__}: {exc}")
        finally:
            self.handles["estimate_sun"].disabled = False

    def _add_vehicle_event(self, event) -> None:
        try:
            config = str(self.handles["vehicle_config"].value).strip() or None
            controller = self.add_vehicle(
                self.handles["asset_folder"].value,
                config,
            )
            self._notify(event, "Vehicle added", controller.vehicle_id)
        except Exception as exc:
            traceback.print_exc()
            self._notify(event, "Vehicle load failed", f"{type(exc).__name__}: {exc}")

    def replace_scene(self, path: str | Path) -> None:
        layer = load_scene(path, self.args.scene_cache_dir)
        indices = _scene_display_indices(
            len(layer.centers),
            int(getattr(self.args, "scene_max_splats", 0)),
            int(self.args.seed),
        )
        if self.scene_handle is not None:
            self.scene_handle.remove()
            self.scene_handle = None
        new_handle = self.server.scene.add_gaussian_splats(
            "/standalone_scene/splats", centers=layer.centers[indices], covariances=layer.covariances[indices],
            rgbs=layer.colors[indices], opacities=layer.opacities[indices], visible=bool(self.handles["scene_visible"].value),
        )
        self.scene_handle = new_handle
        self.scene_path = str(Path(path).expanduser().resolve())
        self.scene_context = resolve_scene_context(self.scene_path)
        self.invalidate_environment_maps()
        print(f"Loaded scene {self.scene_path}: {len(layer.centers[indices]):,}/{layer.total_splats:,} splats", flush=True)

    def clear_scene(self) -> None:
        if self.scene_handle is not None:
            self.scene_handle.remove()
        self.scene_handle = None
        self.scene_path = ""
        self.scene_context = {"name": "empty_scene", "data_root": None, "config_path": None}
        self.invalidate_environment_maps()

    def apply_estimated_sun_angles(self, azimuth: float, elevation: float) -> int:
        azimuth = ((float(azimuth) + 180.0) % 360.0) - 180.0
        elevation = float(np.clip(elevation, -10.0, 89.0))
        active = [controller for controller in self.controllers if not controller._removed]
        with self.server.atomic():
            self.handles["sun_enabled"].value = True
            self.handles["sun_azimuth"].value = azimuth
            self.handles["sun_elevation"].value = elevation
            for controller in active:
                controller._suspend_updates = True
                try:
                    controller.handles["light_sun_enabled"].value = True
                    controller.handles["light_sun_azimuth"].value = azimuth
                    controller.handles["light_sun_elevation"].value = elevation
                    controller._sync_mirrored_controls()
                finally:
                    controller._suspend_updates = False
                controller._sync_lighting_enabled()
        self._shared_lighting_changed("sun_estimate")
        for controller in active:
            if not controller.use_scene_lighting():
                controller.schedule_update("sun_estimate")
        return len(active)

    def _sync_scene_visibility(self) -> None:
        if self.scene_handle is not None:
            self.scene_handle.visible = bool(self.handles["scene_visible"].value)

    def _on_environment_client_connect(self, client) -> None:
        if client.client_id in self._environment_client_ids:
            return
        self._environment_client_ids.add(client.client_id)
        self._environment_client = client

        @client.camera.on_update
        def _camera_updated(_) -> None:
            self._environment_client = client
            self._schedule_environment_camera_update()

        self._schedule_environment_camera_update()

    def _schedule_environment_camera_update(self) -> None:
        if not any(
            controller.environment_map_enabled() or controller.environment_preview_enabled()
            for controller in self.controllers
            if not controller._removed
        ):
            return
        with self._environment_camera_lock:
            self._environment_camera_generation += 1
            generation = self._environment_camera_generation
            if self._environment_camera_timer is not None:
                self._environment_camera_timer.cancel()
            self._environment_camera_timer = threading.Timer(
                0.12, self._apply_environment_camera_update, args=(generation,)
            )
            self._environment_camera_timer.daemon = True
            self._environment_camera_timer.start()

    def _apply_environment_camera_update(self, generation: int) -> None:
        with self._environment_camera_lock:
            if generation != self._environment_camera_generation:
                return
        try:
            for controller in list(self.controllers):
                if not controller._removed and (
                    controller.environment_map_enabled() or controller.environment_preview_enabled()
                ):
                    controller.update()
        finally:
            with self._environment_camera_lock:
                if generation == self._environment_camera_generation:
                    self._environment_camera_timer = None

    def environment_camera_position(self) -> np.ndarray | None:
        clients = self.server.get_clients()
        client = self._environment_client
        if client is None or client.client_id not in clients:
            client = next(iter(clients.values()), None)
            self._environment_client = client
        if client is None:
            return None
        try:
            return np.asarray(client.camera.position, dtype=np.float32)
        except AssertionError:
            return None

    def invalidate_environment_maps(self) -> None:
        self._environment_generation += 1
        for controller in list(self.controllers):
            controller.invalidate_environment_map()

    def capture_environment_map(self, center: np.ndarray, resolution: int) -> EnvironmentMap:
        if self.scene_handle is None:
            raise RuntimeError("Load a scene before enabling the environment map")
        clients = self.server.get_clients()
        client = self._environment_client
        if client is None or client.client_id not in clients:
            client = next(iter(clients.values()), None)
        if client is None:
            raise RuntimeError("Open the Viser page before enabling the environment map")
        center = np.asarray(center, dtype=np.float32)
        resolution = int(np.clip(int(resolution), 16, 512))
        rotations = cube_face_rotations()
        with self._environment_capture_lock:
            visibility = []
            for controller in self.controllers:
                for handle in (
                    controller.splat_handle,
                    controller.contact_handle,
                    controller.extension_handle,
                    getattr(controller, "environment_preview_handle", None),
                ):
                    if handle is not None:
                        visibility.append((handle, bool(handle.visible)))
                        handle.visible = False
            scene_visible = bool(self.scene_handle.visible)
            self.scene_handle.visible = True
            client.flush()
            try:
                faces = [
                    client.get_render(
                        resolution,
                        resolution,
                        wxyz=vt.SO3.from_matrix(rotation).wxyz,
                        position=center,
                        fov=math.pi / 2.0,
                        transport_format="png",
                    )
                    for rotation in rotations
                ]
            finally:
                self.scene_handle.visible = scene_visible
                for handle, visible in visibility:
                    handle.visible = visible
                client.flush()
        return EnvironmentMap.from_faces(faces, rotations)

    def shared_lighting(self) -> LightingState:
        temperature = _ui_to_value(self.handles["env_temperature"].value, "temperature")
        return LightingState(
            environment_intensity=float(self.handles["env_intensity"].value),
            environment_rgb=color_temperature_to_rgb(temperature).tolist(),
            environment_temperature_k=temperature,
            sun_enabled=bool(self.handles["sun_enabled"].value),
            sun_intensity=_ui_to_value(self.handles["sun_intensity"].value, "sun_intensity"),
            sun_rgb=[float(self.handles[key].value) for key in ("sun_r", "sun_g", "sun_b")],
            sun_azimuth_deg=float(self.handles["sun_azimuth"].value),
            sun_elevation_deg=float(self.handles["sun_elevation"].value),
            visibility=float(self.handles["visibility"].value),
        )

    def apply_shared_lighting(self, lighting: LightingState) -> None:
        values = {
            "env_intensity": lighting.environment_intensity,
            "env_temperature": _value_to_ui(lighting.environment_temperature_k, "temperature"),
            "sun_enabled": lighting.sun_enabled, "sun_intensity": _value_to_ui(lighting.sun_intensity, "sun_intensity"),
            "sun_r": lighting.sun_rgb[0], "sun_g": lighting.sun_rgb[1], "sun_b": lighting.sun_rgb[2],
            "sun_azimuth": lighting.sun_azimuth_deg, "sun_elevation": lighting.sun_elevation_deg,
            "visibility": lighting.visibility,
        }
        with self.server.atomic():
            for key, value in values.items():
                self.handles[key].value = value
        self._shared_lighting_changed("reset")

    def _shared_lighting_changed(self, source: str) -> None:
        sun = bool(self.handles["sun_enabled"].value)
        for key in ("sun_intensity", "sun_r", "sun_g", "sun_b", "sun_azimuth", "sun_elevation"):
            self.handles[key].disabled = not sun
        for controller in list(self.controllers):
            if controller.use_scene_lighting():
                controller.schedule_update(source)

    def add_vehicle(self, folder: str | Path, config_path: str | Path | None = None):
        vehicle_id = f"vehicle_{self.next_vehicle_index:03d}"
        self.next_vehicle_index += 1
        asset = load_vehicle_asset(folder)
        base = VehicleState(
            vehicle_id=vehicle_id,
            asset_folder=str(asset.root),
            transform=TransformState(position=[float((len(self.controllers) + 1) * self.args.vehicle_spacing), 0.0, 0.0]),
            environment_map_enabled=bool(getattr(self.args, "enable_environment_map", False)),
            environment_map_resolution=int(getattr(self.args, "environment_map_resolution", 128)),
            projection=copy.deepcopy(asset.projection),
        )
        if config_path:
            selected_config = Path(config_path).expanduser().resolve()
        else:
            candidates = sorted((asset.root / "configs").glob("config*.json"))
            candidates = [path for path in candidates if path.resolve() != asset.canonical_config_path]
            rng = np.random.default_rng(self.args.seed + self.next_vehicle_index)
            selected_config = candidates[int(rng.integers(len(candidates)))] if candidates else asset.canonical_config_path
        initial_payload = read_config(selected_config)
        state, scene_lighting = state_from_config(initial_payload, base)
        state.vehicle_id = vehicle_id
        state.asset_folder = str(asset.root)
        controller = VehicleController(self, asset, state, str(selected_config))
        self.controllers.append(controller)
        if scene_lighting is not None:
            self.apply_shared_lighting(scene_lighting)
        print(f"Loaded {vehicle_id} ({asset.asset_id}): {len(asset.original.centers):,} original / {len(asset.proxy.centers):,} proxy", flush=True)
        return controller

    def remove_vehicle(self, controller: "VehicleController") -> None:
        if controller in self.controllers:
            controller.remove()
            self.controllers.remove(controller)

    def scene_snapshot(self) -> dict[str, Any]:
        return {
            "name": self.scene_context["name"],
            "path": self.scene_path,
            "data_root": self.scene_context.get("data_root"),
        }

    def scene_ply_for_auto_fit(self) -> Path:
        source_value = self.scene_path or str(self.handles["scene_path"].value).strip()
        if not source_value:
            raise ValueError("请先加载普通 Gaussian 场景")
        source = Path(source_value).expanduser().resolve()
        if source.suffix.lower() == ".pth":
            return materialize_pth_scene(source, self.args.scene_cache_dir)
        if source.suffix.lower() != ".ply" or not source.is_file():
            raise ValueError(f"Auto 需要可用的 PLY/PTH 场景: {source}")
        return source


class VehicleController:
    def __init__(self, app: StandaloneViewer, asset: VehicleAsset, state: VehicleState, config_path: str):
        self.app = app
        self.server = app.server
        self.asset = asset
        self.vehicle_id = state.vehicle_id
        self.state = state
        self.config_path = config_path
        self.handles: dict[str, Any] = {}
        self.gui_root = None
        self.splat_handle = None
        self.contact_handle = None
        self.extension_handle = None
        self._timer = None
        self._generation = 0
        self._lock = threading.Lock()
        self._removed = False
        self._suspend_updates = False
        self._auto_fit_thread = None
        self._environment_map_cache = None
        self._environment_map_cache_key = None
        self._environment_map_error = None
        self._environment_preview_cache = None
        self._environment_preview_cache_key = None
        self.environment_preview_handle = None
        self._environment_preview_mesh_key = None
        self._pbr_tensor_cache: dict[Any, Any] = {}
        self._build_gui()
        self.apply_state(state)

    def _build_gui(self) -> None:
        self.gui_root = self.server.gui.add_folder(self.vehicle_id, expand_by_default=False)
        with self.gui_root:
            self.handles["light_sun_intensity"] = self.server.gui.add_slider("太阳光强度", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
            self.handles["light_sun_azimuth"] = self.server.gui.add_slider("太阳方位角", min=-180.0, max=180.0, step=1.0, initial_value=45.0)
            self.handles["light_sun_elevation"] = self.server.gui.add_slider("太阳高度角", min=-10.0, max=89.0, step=1.0, initial_value=35.0)
            self.handles["light_temperature"] = self.server.gui.add_slider("车辆色温", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
            self.handles["saturation"] = self.server.gui.add_slider("饱和度", min=0.0, max=2.0, step=0.01, initial_value=1.0)
            self.handles["ambient_fill"] = self.server.gui.add_slider("亮度", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
            center_button = self.server.gui.add_button("Center orbit on this vehicle")
            self.handles["auto_fit"] = self.server.gui.add_button(
                "Auto 识别车辆参数",
                hint="太阳光强度和亮度固定在绝对 UI 区间 -0.3 到 0.3 内搜索。",
            )
            config_folder = self.server.gui.add_folder("Config", expand_by_default=False)
            advanced_folder = self.server.gui.add_folder("高级", expand_by_default=False)
            with advanced_folder:
                transform_folder = self.server.gui.add_folder("Transform", expand_by_default=False)
            with transform_folder:
                self.handles["x"] = self.server.gui.add_slider("X", min=-100.0, max=100.0, step=0.1, initial_value=0.0)
                self.handles["y"] = self.server.gui.add_slider("Y", min=-100.0, max=100.0, step=0.1, initial_value=0.0)
                self.handles["z"] = self.server.gui.add_slider("Z", min=-10.0, max=10.0, step=0.01, initial_value=0.0)
                for key, label in (("roll", "Roll"), ("pitch", "Pitch"), ("yaw", "Yaw")):
                    self.handles[key] = self.server.gui.add_slider(label, min=-180.0, max=180.0, step=1.0, initial_value=0.0)
                self.handles["scale"] = self.server.gui.add_slider("Scale", min=0.05, max=5.0, step=0.05, initial_value=1.0)
            with advanced_folder:
                material_folder = self.server.gui.add_folder("Material", expand_by_default=False)
            with material_folder:
                self.handles["ambient_fill_advanced"] = self.server.gui.add_slider("亮度", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                self.handles["saturation_advanced"] = self.server.gui.add_slider("饱和度", min=0.0, max=2.0, step=0.01, initial_value=1.0)
                self.handles["roughness"] = self.server.gui.add_slider("Roughness", min=0.02, max=0.98, step=0.01, initial_value=0.4)
                self.handles["reflectance"] = self.server.gui.add_slider("Reflectance", min=0.02, max=0.20, step=0.005, initial_value=0.04)
                self.handles["metallic"] = self.server.gui.add_slider("Metallic", min=0.0, max=1.0, step=0.02, initial_value=0.0)
                self.handles["relight_strength"] = self.server.gui.add_slider("Relight mix", min=0.0, max=1.0, step=0.05, initial_value=1.0)
                self.handles["use_proxy_relighting"] = self.server.gui.add_checkbox(
                    "使用代理重光照",
                    initial_value=True,
                    hint="开启：代理层计算后映射到可见层；关闭：直接显示 PBR 层的朴素着色结果。",
                )
                self.handles["use_asset_material"] = self.server.gui.add_checkbox("Use asset material", initial_value=True)
                self.handles["exposure"] = self.server.gui.add_slider("Exposure", min=0.2, max=3.0, step=0.05, initial_value=1.0)
            with advanced_folder:
                lighting_folder = self.server.gui.add_folder("R3GW Lighting", expand_by_default=False)
            with lighting_folder:
                self.handles["light_temperature_advanced"] = self.server.gui.add_slider("车辆色温", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
                self.handles["light_sun_intensity_advanced"] = self.server.gui.add_slider("太阳光强度", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
                self.handles["light_sun_azimuth_advanced"] = self.server.gui.add_slider("太阳方位角", min=-180.0, max=180.0, step=1.0, initial_value=45.0)
                self.handles["light_sun_elevation_advanced"] = self.server.gui.add_slider("太阳高度角", min=-10.0, max=89.0, step=1.0, initial_value=35.0)
                self.handles["light_intensity"] = self.server.gui.add_slider("Environment intensity", min=0.0, max=4.0, step=0.05, initial_value=1.0)
                self.handles["light_sun_r"] = self.server.gui.add_slider("Sun red", min=0.0, max=3.0, step=0.05, initial_value=1.0)
                self.handles["light_sun_g"] = self.server.gui.add_slider("Sun green", min=0.0, max=3.0, step=0.05, initial_value=1.0)
                self.handles["light_sun_b"] = self.server.gui.add_slider("Sun blue", min=0.0, max=3.0, step=0.05, initial_value=1.0)
                self.handles["use_scene_lighting"] = self.server.gui.add_checkbox("Use scene lighting", initial_value=True)
                self.handles["light_sun_enabled"] = self.server.gui.add_checkbox("Directional sun", initial_value=True)
                self.handles["light_visibility"] = self.server.gui.add_slider("Visibility scale", min=0.0, max=1.0, step=0.05, initial_value=1.0)
                self.handles["environment_map_enabled"] = self.server.gui.add_checkbox(
                    "使用场景环境贴图",
                    initial_value=bool(self.state.environment_map_enabled),
                    hint=(
                        "在车辆中心捕获 scene-only cubemap，并用车辆法线、粗糙度、金属度和当前观察方向计算 IBL。"
                        "关闭时完全回退到原有 SH 环境光预览。"
                    ),
                )
                self.handles["environment_map_resolution"] = self.server.gui.add_dropdown(
                    "环境贴图分辨率",
                    options=("32", "64", "128", "256"),
                    initial_value=str(int(self.state.environment_map_resolution)),
                    hint="每个 cubemap 面的分辨率；车辆位置变化后自动重新捕获。",
                )
                self.handles["environment_map_refresh"] = self.server.gui.add_button("刷新环境贴图")
                preview_folder = self.server.gui.add_folder("Environment-map 预览", expand_by_default=False)
                with preview_folder:
                    self.handles["environment_preview_visible"] = self.server.gui.add_checkbox(
                        "显示环境球", initial_value=bool(self.state.environment_preview_visible)
                    )
                    self.handles["environment_preview_follow_vehicle"] = self.server.gui.add_checkbox(
                        "跟随车辆",
                        initial_value=bool(self.state.environment_preview_follow_vehicle),
                        hint="开启后从车辆中心捕获环境；XYZ 偏移只移动显示球，不改变采样中心。",
                    )
                    for index, axis in enumerate(("x", "y", "z")):
                        self.handles[f"environment_preview_{axis}"] = self.server.gui.add_slider(
                            f"独立采样 {axis.upper()}", min=-100.0, max=100.0, step=0.1,
                            initial_value=float(self.state.environment_preview_position[index]),
                        )
                        self.handles[f"environment_preview_offset_{axis}"] = self.server.gui.add_slider(
                            f"显示偏移 {axis.upper()}", min=-20.0, max=20.0, step=0.1,
                            initial_value=float(self.state.environment_preview_offset[index]),
                        )
            with advanced_folder:
                projection_folder = self.server.gui.add_folder("Projection", expand_by_default=False)
            with projection_folder:
                self.handles["projection_visible"] = self.server.gui.add_checkbox("Show sun projection", initial_value=True)
                self.handles["projection_opacity"] = self.server.gui.add_slider("Overall opacity", min=0.0, max=2.0, step=0.05, initial_value=1.0)
                contact_folder = self.server.gui.add_folder("Contact core", expand_by_default=False)
                extension_folder = self.server.gui.add_folder("Cast extension", expand_by_default=False)
                anchor_folder = self.server.gui.add_folder("Anchor", expand_by_default=False)
                with contact_folder:
                    specs = (
                        ("contact_length", "Length scale", 0.05, 2.0, 0.01),
                        ("contact_width", "Width scale", 0.05, 2.0, 0.01),
                        ("contact_offset_x", "Forward offset (m)", -2.0, 2.0, 0.01),
                        ("contact_offset_y", "Left offset (m)", -2.0, 2.0, 0.01),
                        ("contact_radius", "Corner radius (m)", 0.0, 2.0, 0.01),
                        ("contact_softness", "Edge softness (m)", 0.0, 1.0, 0.01),
                        ("contact_opacity", "Opacity", 0.0, 1.0, 0.01),
                        ("contact_brightness", "Brightness", 0.0, 1.0, 0.01),
                    )
                    for key, label, low, high, step in specs:
                        self.handles[key] = self.server.gui.add_slider(label, min=low, max=high, step=step, initial_value=low)
                with extension_folder:
                    specs = (
                        ("ext_opacity", "Opacity", 0.0, 1.0, 0.01),
                        ("ext_opacity_decay", "Opacity distance decay", 0.0, 1.0, 0.01),
                        ("ext_opacity_exponent", "Opacity distance exponent", 0.05, 8.0, 0.05),
                        ("ext_brightness", "Brightness", 0.0, 1.0, 0.01),
                        ("ext_brightness_white", "Brightness distance to white", 0.0, 1.0, 0.01),
                        ("ext_brightness_exponent", "Brightness distance exponent", 0.05, 8.0, 0.05),
                        ("ext_softness", "Edge softness (m)", 0.0, 1.0, 0.01),
                        ("ext_softness_growth", "Edge softness growth (m)", 0.0, 2.0, 0.01),
                        ("ext_softness_exponent", "Edge softness distance exponent", 0.05, 8.0, 0.05),
                        ("ext_distance", "Distance scale (m)", 0.01, 20.0, 0.05),
                    )
                    for key, label, low, high, step in specs:
                        self.handles[key] = self.server.gui.add_slider(label, min=low, max=high, step=step, initial_value=0.1)
                with anchor_folder:
                    self.handles["anchor_percentile"] = self.server.gui.add_slider("Bottom surface percentile", min=0.0, max=10.0, step=0.1, initial_value=1.0)
                    self.handles["anchor_sigma"] = self.server.gui.add_slider("Surface sigma", min=0.0, max=5.0, step=0.05, initial_value=1.0)
                    self.handles["anchor_z"] = self.server.gui.add_slider("Z offset (m)", min=-0.2, max=0.2, step=0.001, initial_value=0.0)
            with config_folder:
                self.handles["config_path"] = self.server.gui.add_text("Config path", initial_value=self.config_path)
                save_button = self.server.gui.add_button("Save vehicle + lighting config")
                load_button = self.server.gui.add_button("Load config")
                delete_button = self.server.gui.add_button("Delete vehicle")
                self.handles["visible"] = self.server.gui.add_checkbox("Visible", initial_value=True)
                self.handles["mode"] = self.server.gui.add_dropdown("Display", options=DISPLAY_MODES, initial_value="Relight Original")

        mirrored_controls = self._mirrored_controls()
        mirrored_keys = set(mirrored_controls) | set(mirrored_controls.values())
        for key, handle in self.handles.items():
            if (
                key not in {"config_path", "auto_fit"}
                and key != "environment_map_refresh"
                and key not in mirrored_keys
            ):
                handle.on_update(lambda _, source=key: self._control_changed(source))
        for canonical, advanced in mirrored_controls.items():
            self.handles[canonical].on_update(
                lambda _, source=canonical, target=advanced: self._mirrored_control_changed(source, target, source)
            )
            self.handles[advanced].on_update(
                lambda _, source=advanced, target=canonical, canonical_key=canonical:
                    self._mirrored_control_changed(source, target, canonical_key)
            )
        self._sync_mirrored_controls()
        save_button.on_click(lambda event: self._save_event(event))
        load_button.on_click(lambda event: self._load_event(event))
        delete_button.on_click(lambda _: self.app.remove_vehicle(self))
        center_button.on_click(self._center_orbit_on_vehicle)
        self.handles["auto_fit"].on_click(self._auto_fit_event)
        self.handles["environment_map_refresh"].on_click(self._refresh_environment_map)

    @staticmethod
    def _mirrored_controls() -> dict[str, str]:
        return {
            "light_sun_intensity": "light_sun_intensity_advanced",
            "light_sun_azimuth": "light_sun_azimuth_advanced",
            "light_sun_elevation": "light_sun_elevation_advanced",
            "light_temperature": "light_temperature_advanced",
            "saturation": "saturation_advanced",
            "ambient_fill": "ambient_fill_advanced",
        }

    def _sync_mirrored_controls(self) -> None:
        self._suspend_updates = True
        try:
            with self.server.atomic():
                for canonical, advanced in self._mirrored_controls().items():
                    self.handles[advanced].value = self.handles[canonical].value
        finally:
            self._suspend_updates = False

    def _mirrored_control_changed(self, source: str, target: str, canonical: str) -> None:
        if self._suspend_updates or self._removed:
            return
        self._suspend_updates = True
        try:
            self.handles[target].value = self.handles[source].value
            if canonical.startswith("light_"):
                self.handles["use_scene_lighting"].value = False
            if canonical.startswith("light_sun_"):
                self.handles["light_sun_enabled"].value = True
        finally:
            self._suspend_updates = False
        self._control_changed(canonical)

    def _control_changed(self, source: str) -> None:
        if self._suspend_updates or self._removed:
            return
        if source in {
            "x", "y", "z", "roll", "pitch", "yaw", "scale",
            "environment_map_enabled", "environment_map_resolution",
            "environment_preview_visible", "environment_preview_follow_vehicle",
            "environment_preview_x", "environment_preview_y", "environment_preview_z",
        }:
            self.invalidate_environment_map()
        self._sync_lighting_enabled()
        self.schedule_update(source)

    def _sync_lighting_enabled(self) -> None:
        local = not bool(self.handles["use_scene_lighting"].value)
        sun = bool(self.handles["light_sun_enabled"].value)
        for key in ("light_intensity", "light_sun_enabled", "light_visibility"):
            self.handles[key].disabled = not local
        for key in ("light_sun_r", "light_sun_g", "light_sun_b"):
            self.handles[key].disabled = (not local) or (not sun)
        for key in (
            "light_temperature", "light_temperature_advanced", "light_sun_intensity", "light_sun_intensity_advanced",
            "light_sun_azimuth", "light_sun_azimuth_advanced", "light_sun_elevation", "light_sun_elevation_advanced",
        ):
            self.handles[key].disabled = False
        environment_enabled = self.environment_map_enabled()
        preview_enabled = bool(self.handles["environment_preview_visible"].value)
        follow_vehicle = bool(self.handles["environment_preview_follow_vehicle"].value)
        self.handles["environment_map_resolution"].disabled = not (environment_enabled or preview_enabled)
        self.handles["environment_map_refresh"].disabled = not (environment_enabled or preview_enabled)
        self.handles["environment_preview_follow_vehicle"].disabled = not preview_enabled
        for axis in ("x", "y", "z"):
            self.handles[f"environment_preview_{axis}"].disabled = (not preview_enabled) or follow_vehicle
            self.handles[f"environment_preview_offset_{axis}"].disabled = (not preview_enabled) or (not follow_vehicle)

    def _center_orbit_on_vehicle(self, event=None) -> None:
        mode = str(self.handles["mode"].value)
        layer = self.asset.original if mode in {"Relight Original", "Original SH"} else self.asset.proxy
        transform = self.transform()
        local_center = 0.5 * (np.min(layer.centers, axis=0) + np.max(layer.centers, axis=0))
        local_center *= transform.scale
        quaternion = euler_to_wxyz(**{f"{key}_deg": value for key, value in transform.rotation_deg.items()})
        rotation = quaternion_to_rotation(quaternion[None, :])[0]
        center = rotation @ local_center + np.asarray(transform.position, dtype=np.float32)
        for client in self.server.get_clients().values():
            client.camera.look_at = center
        self.app._notify(event, "Vehicle orbit center updated", f"{self.vehicle_id}: {center.round(2).tolist()}")

    def schedule_update(self, source: str = "") -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(0.12, self._run_scheduled, args=(generation,))
            self._timer.daemon = True
            self._timer.start()

    def _run_scheduled(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
        try:
            self.update()
        except Exception:
            traceback.print_exc()

    def use_scene_lighting(self) -> bool:
        return bool(self.handles["use_scene_lighting"].value)

    def lighting(self) -> LightingState:
        if self.use_scene_lighting():
            return self.app.shared_lighting()
        temperature = _ui_to_value(self.handles["light_temperature"].value, "temperature")
        return LightingState(
            environment_intensity=float(self.handles["light_intensity"].value),
            environment_rgb=color_temperature_to_rgb(temperature).tolist(),
            environment_temperature_k=temperature,
            sun_enabled=bool(self.handles["light_sun_enabled"].value),
            sun_intensity=_ui_to_value(self.handles["light_sun_intensity"].value, "sun_intensity"),
            sun_rgb=[float(self.handles[key].value) for key in ("light_sun_r", "light_sun_g", "light_sun_b")],
            sun_azimuth_deg=float(self.handles["light_sun_azimuth"].value),
            sun_elevation_deg=float(self.handles["light_sun_elevation"].value),
            visibility=float(self.handles["light_visibility"].value),
        )

    def material(self) -> MaterialState:
        values = {key: self.handles[key].value for key in MaterialState.__dataclass_fields__}
        values["saturation"] = _ui_to_value(values["saturation"], "saturation")
        values["ambient_fill"] = _ui_to_value(values["ambient_fill"], "brightness")
        return MaterialState(**values)

    def environment_map_enabled(self) -> bool:
        return bool(self.handles.get("environment_map_enabled")) and bool(
            self.handles["environment_map_enabled"].value
        )

    def environment_preview_enabled(self) -> bool:
        return bool(self.handles.get("environment_preview_visible")) and bool(
            self.handles["environment_preview_visible"].value
        )

    def invalidate_environment_map(self) -> None:
        self._environment_map_cache = None
        self._environment_map_cache_key = None
        self._environment_map_error = None
        self._environment_preview_cache = None
        self._environment_preview_cache_key = None

    def _vehicle_environment_center(self, rotation: np.ndarray, transform: TransformState) -> np.ndarray:
        local_min = np.min(self.asset.proxy.centers, axis=0)
        local_max = np.max(self.asset.proxy.centers, axis=0)
        local_center = 0.5 * (local_min + local_max) * transform.scale
        return np.asarray(transform.position, dtype=np.float32) + rotation @ local_center

    def _remove_environment_preview(self) -> None:
        if self.environment_preview_handle is not None:
            self.environment_preview_handle.remove()
            self.environment_preview_handle = None
        self._environment_preview_mesh_key = None

    def _update_environment_preview(
        self,
        environment_map: EnvironmentMap | None,
        cache_key,
        vehicle_center: np.ndarray,
        sample_center: np.ndarray,
    ) -> None:
        if not self.environment_preview_enabled():
            self._remove_environment_preview()
            return
        if bool(self.handles["environment_preview_follow_vehicle"].value):
            offset = np.asarray(
                [
                    float(self.handles[f"environment_preview_offset_{axis}"].value)
                    for axis in ("x", "y", "z")
                ],
                dtype=np.float32,
            )
            display_position = vehicle_center + offset
        else:
            display_position = sample_center
        if environment_map is None:
            if self.environment_preview_handle is not None:
                self.environment_preview_handle.position = display_position
            return
        if self.environment_preview_handle is None or self._environment_preview_mesh_key != cache_key:
            self._remove_environment_preview()
            self.environment_preview_handle = self.server.scene.add_mesh_trimesh(
                f"/vehicles/{self.vehicle_id}/environment_map_preview",
                textured_environment_sphere(environment_map),
                scale=1.5,
                position=display_position,
                visible=True,
                cast_shadow=False,
                receive_shadow=False,
            )
            self._environment_preview_mesh_key = cache_key
        else:
            self.environment_preview_handle.position = display_position
            self.environment_preview_handle.visible = True

    def _environment_map_for(self, rotation: np.ndarray, transform: TransformState):
        if not self.environment_map_enabled() and not self.environment_preview_enabled():
            self._remove_environment_preview()
            return None
        vehicle_center = self._vehicle_environment_center(rotation, transform)
        resolution = int(self.handles["environment_map_resolution"].value)
        vehicle_cache_key = (
            int(self.app._environment_generation),
            resolution,
            tuple(np.round(vehicle_center, 4).tolist()),
        )
        vehicle_environment = None
        if self.environment_map_enabled():
            if self._environment_map_cache is not None and vehicle_cache_key == self._environment_map_cache_key:
                vehicle_environment = self._environment_map_cache
            else:
                try:
                    vehicle_environment = self.app.capture_environment_map(vehicle_center, resolution)
                    self._environment_map_cache = vehicle_environment
                    self._environment_map_cache_key = vehicle_cache_key
                    self._environment_map_error = None
                    print(
                        f"[environment-map] captured {self.vehicle_id} at {vehicle_center.round(3).tolist()} "
                        f"({resolution}px x 6)", flush=True,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message != self._environment_map_error:
                        print(
                            f"[environment-map] {self.vehicle_id} capture failed; using legacy SH: {message}",
                            flush=True,
                        )
                    self._environment_map_error = message

        if self.environment_preview_enabled():
            follow_vehicle = bool(self.handles["environment_preview_follow_vehicle"].value)
            preview_center = vehicle_center if follow_vehicle else np.asarray(
                [float(self.handles[f"environment_preview_{axis}"].value) for axis in ("x", "y", "z")],
                dtype=np.float32,
            )
            preview_key = (
                int(self.app._environment_generation),
                resolution,
                tuple(np.round(preview_center, 4).tolist()),
            )
            preview_environment = vehicle_environment if follow_vehicle and vehicle_environment is not None else None
            if preview_environment is None and self._environment_preview_cache_key == preview_key:
                preview_environment = self._environment_preview_cache
            if preview_environment is None:
                try:
                    preview_environment = self.app.capture_environment_map(preview_center, resolution)
                    self._environment_preview_cache = preview_environment
                    self._environment_preview_cache_key = preview_key
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message != self._environment_map_error:
                        print(f"[environment-map-preview] {self.vehicle_id} capture failed: {message}", flush=True)
                    self._environment_map_error = message
            self._update_environment_preview(preview_environment, preview_key, vehicle_center, preview_center)
        else:
            self._remove_environment_preview()
        return vehicle_environment

    def _refresh_environment_map(self, event=None) -> None:
        self.invalidate_environment_map()
        try:
            self.update()
            active_cache = (
                self._environment_map_cache
                if self.environment_map_enabled()
                else self._environment_preview_cache
            )
            if active_cache is None:
                raise RuntimeError(self._environment_map_error or "environment map is unavailable")
            self.app._notify(
                event,
                "环境贴图已刷新",
                f"{self.vehicle_id}: {self.handles['environment_map_resolution'].value}px x 6",
            )
        except Exception as exc:
            self.app._notify(event, "环境贴图刷新失败", f"{type(exc).__name__}: {exc}")

    def auto_fit_template(self) -> dict[str, Any]:
        material = self.material()
        lighting = self.lighting()
        state = self.snapshot().to_dict()
        state.update(material.to_dict())
        state["vehicle_lighting"] = {
            "use_scene_lighting": False,
            "intensity": lighting.environment_intensity,
            "environment_temperature": lighting.environment_temperature_k,
            "red": lighting.environment_rgb[0],
            "green": lighting.environment_rgb[1],
            "blue": lighting.environment_rgb[2],
            "sun_enabled": lighting.sun_enabled,
            "sun_intensity": lighting.sun_intensity,
            "sun_color_rgb": lighting.sun_rgb,
            "sun_azimuth": lighting.sun_azimuth_deg,
            "sun_elevation": lighting.sun_elevation_deg,
            "visibility": lighting.visibility,
        }
        return {
            "schema_version": 1,
            "scene": self.app.scene_snapshot(),
            "pbr_asset": copy.deepcopy(self.asset.canonical_config),
            "vehicle": state,
        }

    def _auto_fit_event(self, event) -> None:
        if self._auto_fit_thread is not None and self._auto_fit_thread.is_alive():
            self.app._notify(event, "Auto 识别进行中", f"{self.vehicle_id} 正在计算")
            return
        self.handles["auto_fit"].disabled = True
        self.app._notify(event, "Auto 识别已启动", "正在匹配太阳光强度、亮度和车辆色温")
        self._auto_fit_thread = threading.Thread(target=self._run_auto_fit, args=(event,), daemon=True)
        self._auto_fit_thread.start()

    def _run_auto_fit(self, event) -> None:
        try:
            result = run_vehicle_auto_fit(
                scene_ply=self.app.scene_ply_for_auto_fit(),
                asset_dir=self.asset.root,
                template_payload=self.auto_fit_template(),
                config_path=str(self.app.handles["auto_fit_config"].value).strip() or None,
                timeout=float(getattr(self.app.args, "auto_fit_timeout", 600.0)),
                device=getattr(getattr(self.app, "compute_device", None), "value", "auto"),
            )
            temperature = float(result["environment_temperature_k"])
            self._suspend_updates = True
            try:
                with self.server.atomic():
                    self.handles["use_scene_lighting"].value = False
                    self.handles["light_sun_enabled"].value = True
                    self.handles["light_sun_intensity"].value = _value_to_ui(
                        result["sun_intensity"], "sun_intensity"
                    )
                    self.handles["ambient_fill"].value = _value_to_ui(
                        result["ambient_fill"], "brightness"
                    )
                    self.handles["light_temperature"].value = _value_to_ui(
                        temperature, "temperature"
                    )
            finally:
                self._suspend_updates = False
            self._sync_mirrored_controls()
            self._sync_lighting_enabled()
            self.update()
            self.app._notify(
                event,
                "Auto 识别完成",
                f"太阳光强度、亮度和车辆色温已更新；相对原始 DC 指标变化 {result['improvement_percent']:.2f}%，结果 {result['output_dir']}",
            )
        except Exception as exc:
            traceback.print_exc()
            self.app._notify(event, "Auto 识别未应用", f"{type(exc).__name__}: {exc}")
        finally:
            if not self._removed:
                self.handles["auto_fit"].disabled = False

    def transform(self) -> TransformState:
        return TransformState(
            position=[float(self.handles[key].value) for key in ("x", "y", "z")],
            rotation_deg={key: float(self.handles[key].value) for key in ("roll", "pitch", "yaw")},
            scale=float(self.handles["scale"].value),
        )

    def projection(self) -> dict[str, Any]:
        result = copy.deepcopy(self.asset.projection)
        result.update({"enabled": True, "dynamic": True, "runtime": "receiver_space_mask_v1", "shape_model": "procedural-mask-contact-silhouette-v4", "composition": "receiver_alpha_blend_to_grayscale_v1"})
        result["contact"].update({
            "length_scale": float(self.handles["contact_length"].value),
            "width_scale": float(self.handles["contact_width"].value),
            "offset_xy_m": [float(self.handles["contact_offset_x"].value), float(self.handles["contact_offset_y"].value)],
            "corner_radius_m": float(self.handles["contact_radius"].value),
            "edge_softness_m": float(self.handles["contact_softness"].value),
            "opacity": float(self.handles["contact_opacity"].value),
            "brightness": float(self.handles["contact_brightness"].value),
        })
        result["extension"].update({
            "opacity": float(self.handles["ext_opacity"].value),
            "opacity_distance_decay": float(self.handles["ext_opacity_decay"].value),
            "opacity_distance_exponent": float(self.handles["ext_opacity_exponent"].value),
            "brightness": float(self.handles["ext_brightness"].value),
            "brightness_distance_to_white": float(self.handles["ext_brightness_white"].value),
            "brightness_distance_exponent": float(self.handles["ext_brightness_exponent"].value),
            "edge_softness_m": float(self.handles["ext_softness"].value),
            "edge_softness_distance_growth_m": float(self.handles["ext_softness_growth"].value),
            "edge_softness_distance_exponent": float(self.handles["ext_softness_exponent"].value),
            "distance_scale_m": float(self.handles["ext_distance"].value),
        })
        result["anchor"].update({
            "mode": "gaussian_bottom_surface",
            "bottom_surface_percentile": float(self.handles["anchor_percentile"].value),
            "surface_sigma": float(self.handles["anchor_sigma"].value),
            "z_offset_m": float(self.handles["anchor_z"].value),
        })
        return result

    def snapshot(self) -> VehicleState:
        projection = self.projection()
        projection.pop("_fixed_ground_z", None)
        projection.pop("_fixed_anchor", None)
        return VehicleState(
            vehicle_id=self.vehicle_id, asset_folder=str(self.asset.root),
            visible=bool(self.handles["visible"].value), display_mode=str(self.handles["mode"].value),
            transform=self.transform(), material=self.material(),
            use_proxy_relighting=bool(self.handles["use_proxy_relighting"].value),
            use_scene_lighting=self.use_scene_lighting(),
            environment_map_enabled=self.environment_map_enabled(),
            environment_map_resolution=int(self.handles["environment_map_resolution"].value),
            environment_preview_visible=bool(self.handles["environment_preview_visible"].value),
            environment_preview_follow_vehicle=bool(
                self.handles["environment_preview_follow_vehicle"].value
            ),
            environment_preview_position=[
                float(self.handles[f"environment_preview_{axis}"].value) for axis in ("x", "y", "z")
            ],
            environment_preview_offset=[
                float(self.handles[f"environment_preview_offset_{axis}"].value) for axis in ("x", "y", "z")
            ],
            lighting=self.lighting() if not self.use_scene_lighting() else LightingState.from_dict({
                "environment_intensity": float(self.handles["light_intensity"].value),
                "environment_temperature_k": _ui_to_value(self.handles["light_temperature"].value, "temperature"),
                "environment_rgb": color_temperature_to_rgb(_ui_to_value(self.handles["light_temperature"].value, "temperature")).tolist(),
                "sun_enabled": bool(self.handles["light_sun_enabled"].value),
                "sun_intensity": _ui_to_value(self.handles["light_sun_intensity"].value, "sun_intensity"),
                "sun_rgb": [float(self.handles[key].value) for key in ("light_sun_r", "light_sun_g", "light_sun_b")],
                "sun_azimuth_deg": float(self.handles["light_sun_azimuth"].value),
                "sun_elevation_deg": float(self.handles["light_sun_elevation"].value),
                "visibility": float(self.handles["light_visibility"].value),
            }),
            projection_visible=bool(self.handles["projection_visible"].value),
            projection_opacity=float(self.handles["projection_opacity"].value), projection=projection,
        )

    def apply_state(self, state: VehicleState) -> None:
        p = state.projection or self.asset.projection
        c, e, a = p["contact"], p["extension"], p["anchor"]
        values = {
            "visible": state.visible, "mode": state.display_mode,
            "x": state.transform.position[0], "y": state.transform.position[1], "z": state.transform.position[2],
            **state.transform.rotation_deg, "scale": state.transform.scale,
            **state.material.to_dict(),
            "use_proxy_relighting": state.use_proxy_relighting,
            "use_scene_lighting": state.use_scene_lighting,
            "light_intensity": state.lighting.environment_intensity,
            "light_temperature": _value_to_ui(state.lighting.environment_temperature_k, "temperature"),
            "light_sun_enabled": state.lighting.sun_enabled, "light_sun_intensity": _value_to_ui(state.lighting.sun_intensity, "sun_intensity"),
            "light_sun_r": state.lighting.sun_rgb[0], "light_sun_g": state.lighting.sun_rgb[1], "light_sun_b": state.lighting.sun_rgb[2],
            "light_sun_azimuth": state.lighting.sun_azimuth_deg, "light_sun_elevation": state.lighting.sun_elevation_deg,
            "light_visibility": state.lighting.visibility,
            "environment_map_enabled": state.environment_map_enabled,
            "environment_map_resolution": str(state.environment_map_resolution),
            "environment_preview_visible": state.environment_preview_visible,
            "environment_preview_follow_vehicle": state.environment_preview_follow_vehicle,
            "projection_visible": state.projection_visible, "projection_opacity": state.projection_opacity,
            "contact_length": c["length_scale"], "contact_width": c["width_scale"],
            "contact_offset_x": c["offset_xy_m"][0], "contact_offset_y": c["offset_xy_m"][1],
            "contact_radius": c["corner_radius_m"], "contact_softness": c["edge_softness_m"],
            "contact_opacity": c["opacity"], "contact_brightness": c["brightness"],
            "ext_opacity": e["opacity"], "ext_opacity_decay": e["opacity_distance_decay"],
            "ext_opacity_exponent": e["opacity_distance_exponent"], "ext_brightness": e["brightness"],
            "ext_brightness_white": e["brightness_distance_to_white"], "ext_brightness_exponent": e["brightness_distance_exponent"],
            "ext_softness": e["edge_softness_m"], "ext_softness_growth": e["edge_softness_distance_growth_m"],
            "ext_softness_exponent": e["edge_softness_distance_exponent"], "ext_distance": e["distance_scale_m"],
            "anchor_percentile": a["bottom_surface_percentile"], "anchor_sigma": a["surface_sigma"], "anchor_z": a["z_offset_m"],
        }
        values.update({
            f"environment_preview_{axis}": float(state.environment_preview_position[index])
            for index, axis in enumerate(("x", "y", "z"))
        })
        values.update({
            f"environment_preview_offset_{axis}": float(state.environment_preview_offset[index])
            for index, axis in enumerate(("x", "y", "z"))
        })
        values["saturation"] = _value_to_ui(state.material.saturation, "saturation")
        values["ambient_fill"] = _value_to_ui(state.material.ambient_fill, "brightness")
        with self.server.atomic():
            for key, value in values.items():
                self.handles[key].value = value
        self._sync_mirrored_controls()
        self._sync_lighting_enabled()
        self.invalidate_environment_map()
        self.update()

    def _effective_local_lighting(self, quaternion: np.ndarray) -> tuple[LightingState, np.ndarray]:
        lighting = self.lighting()
        world_direction = sun_direction(lighting.sun_azimuth_deg, lighting.sun_elevation_deg)
        rotation = quaternion_to_rotation(quaternion[None, :])[0]
        local_direction = normalize((rotation.T @ world_direction).reshape(1, 3))[0]
        local = copy.deepcopy(lighting)
        local.sun_azimuth_deg = math.degrees(math.atan2(float(local_direction[1]), float(local_direction[0])))
        local.sun_elevation_deg = math.degrees(math.asin(float(np.clip(local_direction[2], -1.0, 1.0))))
        return local, local_direction

    def update(self) -> None:
        if self._removed:
            return
        transform = self.transform()
        quaternion = euler_to_wxyz(**{f"{key}_deg": value for key, value in transform.rotation_deg.items()})
        rotation = quaternion_to_rotation(quaternion[None, :])[0]
        environment_map = self._environment_map_for(rotation, transform)
        visible = bool(self.handles["visible"].value)
        if not visible:
            for handle in (self.splat_handle, self.contact_handle, self.extension_handle):
                if handle is not None:
                    handle.visible = False
            return
        local_lighting, local_sun = self._effective_local_lighting(quaternion)
        if environment_map is None or not self.environment_map_enabled():
            layer, colors = shade_vehicle(
                self.asset, self.material(), local_lighting,
                mode=str(self.handles["mode"].value),
                use_proxy_relighting=bool(self.handles["use_proxy_relighting"].value),
                device=self.app.compute_device.value,
                tensor_cache=self._pbr_tensor_cache,
            )
        else:
            world_normals = normalize(self.asset.proxy.normals @ rotation.T)
            world_centers = (
                self.asset.proxy.centers * transform.scale
            ) @ rotation.T + np.asarray(transform.position, dtype=np.float32)[None, :]
            camera_position = self.app.environment_camera_position()
            if camera_position is None:
                fallback_view = normalize(
                    np.asarray([[0.0, -1.0, 0.6]], dtype=np.float32) @ rotation.T
                )[0]
                view_directions = np.broadcast_to(fallback_view, world_normals.shape)
            else:
                view_directions = normalize(camera_position[None, :] - world_centers)
            layer, colors = shade_vehicle(
                self.asset,
                self.material(),
                self.lighting(),
                mode=str(self.handles["mode"].value),
                environment_map=environment_map,
                world_normals=world_normals,
                view_directions=view_directions,
                use_proxy_relighting=bool(self.handles["use_proxy_relighting"].value),
                device=self.app.compute_device.value,
                tensor_cache=self._pbr_tensor_cache,
            )
        centers = np.ascontiguousarray(layer.centers * transform.scale, dtype=np.float32)
        covariances = np.ascontiguousarray(layer.covariances * transform.scale ** 2, dtype=np.float32)
        position = np.asarray(transform.position, dtype=np.float32)
        with self.server.atomic():
            if self.splat_handle is None:
                self.splat_handle = self.server.scene.add_gaussian_splats(
                    f"/vehicles/{self.vehicle_id}/splats", centers=centers, covariances=covariances,
                    rgbs=colors, opacities=layer.opacities, wxyz=quaternion, position=position,
                )
            else:
                self.splat_handle.buffer = pack_gaussian_buffer(centers, covariances, colors, layer.opacities)
                self.splat_handle.wxyz = quaternion
                self.splat_handle.position = position
                self.splat_handle.visible = True
        self._update_projection(quaternion, position, transform.scale, local_lighting, local_sun)

    def _update_projection(self, quaternion, position, scale, lighting, local_sun) -> None:
        enabled = bool(self.handles["projection_visible"].value) and lighting.sun_enabled and local_sun[2] >= math.sin(math.radians(1.0))
        if not enabled:
            for handle in (self.contact_handle, self.extension_handle):
                if handle is not None:
                    handle.visible = False
            return
        result = build_projection_masks(self.asset.proxy, local_sun, self.projection())
        opacity_scale = float(self.handles["projection_opacity"].value) * lighting.visibility
        rotation = quaternion_to_rotation(quaternion[None, :])[0]
        items = (("extension", 0.001), ("contact", 0.002))
        with self.server.atomic():
            for name, z_bias in items:
                item = result[name]
                rgba = item["rgba"].copy()
                rgba[..., 3] = np.clip(rgba[..., 3].astype(np.float32) * opacity_scale, 0, 255).astype(np.uint8)
                size = np.asarray(item["size_xy"], dtype=np.float32) * scale
                center = np.asarray([item["center_xy"][0], item["center_xy"][1], result["ground_z"] + z_bias], dtype=np.float32) * scale
                world_position = position + rotation @ center
                glb = rgba_plane_glb(rgba, float(size[0]), float(size[1]))
                attribute = f"{name}_handle"
                handle = getattr(self, attribute)
                if handle is None:
                    handle = self.server.scene.add_glb(
                        f"/vehicles/{self.vehicle_id}/projection/{name}", glb_data=glb,
                        wxyz=quaternion, position=world_position, cast_shadow=False, receive_shadow=False,
                    )
                    setattr(self, attribute, handle)
                else:
                    handle.glb_data = glb
                    handle.wxyz = quaternion
                    handle.position = world_position
                    handle.visible = True

    def _save_event(self, event) -> None:
        try:
            path = next_config_path(self.asset.root, self.app.scene_snapshot()["name"], self.asset.asset_id)
            save_viewer_config(path, self.app.scene_snapshot(), self.app.shared_lighting(), self.snapshot(), self.asset.canonical_config)
            self.handles["config_path"].value = str(path)
            self.app._notify(event, "Config saved", str(path))
            print(f"Saved config: {path}", flush=True)
        except Exception as exc:
            traceback.print_exc()
            self.app._notify(event, "Config save failed", f"{type(exc).__name__}: {exc}")

    def _load_event(self, event) -> None:
        try:
            payload = read_config(self.handles["config_path"].value)
            state, scene_lighting = state_from_config(payload, self.snapshot())
            if Path(state.asset_folder).resolve() != self.asset.root.resolve():
                raise ValueError("Config belongs to a different asset folder; add that asset as a new vehicle")
            self.apply_state(state)
            if scene_lighting is not None:
                self.app.apply_shared_lighting(scene_lighting)
            self.app._notify(event, "Config loaded", self.handles["config_path"].value)
        except Exception as exc:
            traceback.print_exc()
            self.app._notify(event, "Config load failed", f"{type(exc).__name__}: {exc}")

    def remove(self) -> None:
        self._removed = True
        if self._timer is not None:
            self._timer.cancel()
        for handle in (
            self.splat_handle,
            self.contact_handle,
            self.extension_handle,
            self.environment_preview_handle,
        ):
            if handle is not None:
                handle.remove()
        if self.gui_root is not None:
            self.gui_root.remove()
