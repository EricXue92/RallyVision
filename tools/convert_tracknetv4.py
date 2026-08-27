"""TrackNetV4 权重转换：上游 Keras `.keras` -> 本项目 PyTorch `.pt`。

上游 https://github.com/TrackNetV4/TrackNetV4（MIT License）发布的是 Keras 3
的 `.keras` checkpoint，而本项目全栈 torch（MPS）。为了不把 TensorFlow/Keras
拖进 worker 运行时依赖，权重在这里**离线**转一次：`.keras` 本质是个 zip，里面
的 `model.weights.h5` 是纯数组，用 h5py 直接读即可，全程不需要 TF。

用法 / usage：

    uv run --with h5py tools/convert_tracknetv4.py \
        --keras ~/Desktop/new_tennis/best_model_V1_NF_RIO_10u_e17.keras \
        --out weights/tracknet_v4_typeA.pt

融合类型（Type A / Type B / 无融合的 V2 基线）从 `.keras` 的 config.json 里的
自定义层类名自动识别，也可以 `--fusion` 覆盖。

搬运规则（对不上就是结构错了，脚本会直接报错而不是静默跳过）：
- Conv2D  kernel (kh, kw, in, out) -> torch (out, in, kh, kw)
- BatchNormalization vars 顺序 = [gamma, beta, moving_mean, moving_variance]，
  长度是该层的 **W**（不是通道数，原因见 tracknet_v4_ball.py 模块 docstring）
- MotionPromptLayer vars 顺序 = [a, b]（两个标量）
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_analysis.detection.tracknet_v4_ball import (  # noqa: E402
    MODEL_INPUT_HEIGHT,
    MODEL_INPUT_WIDTH,
    get_tracknet_v4_class,
)

_NUM_CONV = 18
_NUM_BN = 17

# .keras 里保存的自定义层类名 -> 本项目的 fusion 标识。注意这三个权重是用
# 上游较早版本的代码训练的，类名叫 MotionIncorporationLayerV1/V2，而上游
# 当前 master 的源码里已改名成 FusionLayerTypeA/TypeB——同一个东西。
_FUSION_CLASS_TO_TYPE = {
    "MotionIncorporationLayerV1": "A",
    "FusionLayerTypeA": "A",
    "MotionIncorporationLayerV2": "B",
    "FusionLayerTypeB": "B",
}


def _keras_layer_name(prefix, index):
    """Keras 的自动命名：第一个层没有序号后缀，之后从 _1 开始。"""
    return prefix if index == 0 else f"{prefix}_{index}"


def detect_fusion_type(archive):
    config = json.loads(archive.read("config.json"))
    class_names = {layer["class_name"] for layer in config["config"]["layers"]}
    for class_name, fusion in _FUSION_CLASS_TO_TYPE.items():
        if class_name in class_names:
            return fusion
    return "none"


def build_state_dict(weights_file, fusion):
    layers = weights_file["layers"]
    state_dict = {}

    for index in range(_NUM_CONV):
        name = _keras_layer_name("conv2d", index)
        variables = layers[name]["vars"]
        kernel = np.asarray(variables["0"])  # (kh, kw, in, out)
        bias = np.asarray(variables["1"])
        state_dict[f"conv{index}.weight"] = torch.from_numpy(
            np.ascontiguousarray(kernel.transpose(3, 2, 0, 1))
        )
        state_dict[f"conv{index}.bias"] = torch.from_numpy(bias)

    for index in range(_NUM_BN):
        name = _keras_layer_name("batch_normalization", index)
        variables = layers[name]["vars"]
        state_dict[f"bn{index}.gamma"] = torch.from_numpy(np.asarray(variables["0"]))
        state_dict[f"bn{index}.beta"] = torch.from_numpy(np.asarray(variables["1"]))
        state_dict[f"bn{index}.moving_mean"] = torch.from_numpy(np.asarray(variables["2"]))
        state_dict[f"bn{index}.moving_variance"] = torch.from_numpy(np.asarray(variables["3"]))

    if fusion in ("A", "B"):
        variables = layers["motion_prompt_layer"]["vars"]
        state_dict["motion.a"] = torch.tensor(float(np.asarray(variables["0"])))
        state_dict["motion.b"] = torch.tensor(float(np.asarray(variables["1"])))

    return state_dict


def convert(keras_path, out_path, fusion=None, input_height=MODEL_INPUT_HEIGHT,
            input_width=MODEL_INPUT_WIDTH):
    import h5py

    archive = zipfile.ZipFile(keras_path)
    detected = detect_fusion_type(archive)
    fusion = fusion or detected
    if fusion != detected:
        print(f"[warn] --fusion={fusion} 覆盖了自动识别结果 {detected}")

    with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights_file:
        state_dict = build_state_dict(weights_file, fusion)

    # 用真结构 load 一次做形状体检：任何 missing/unexpected/形状不符都会抛，
    # 不允许 strict=False 静默放过（权重搬错了后面只会表现为「检测变差」，
    # 极难回溯到这一步）。
    model = get_tracknet_v4_class()(fusion=fusion, input_height=input_height, input_width=input_width)
    model.load_state_dict(state_dict, strict=True)

    payload = {
        "state_dict": state_dict,
        "fusion": fusion,
        "input_height": input_height,
        "input_width": input_width,
        "source": os.path.basename(keras_path),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(payload, out_path)
    total = sum(v.numel() for v in state_dict.values())
    print(f"[ok] {keras_path} -> {out_path} (fusion={fusion}, {total:,} params)")


def main():
    parser = argparse.ArgumentParser(description="Convert TrackNetV4 .keras weights to PyTorch .pt")
    parser.add_argument("--keras", required=True, help="上游 .keras checkpoint 路径")
    parser.add_argument("--out", required=True, help="输出 .pt 路径")
    parser.add_argument("--fusion", choices=["A", "B", "none"], default=None,
                        help="覆盖自动识别的融合类型（一般不需要）")
    args = parser.parse_args()
    convert(args.keras, args.out, fusion=args.fusion)


if __name__ == "__main__":
    main()
