# 使用示例

本目录包含 YOLO-Go 的使用示例代码。

## 示例文件

- `test_simple.go` - 基本使用示例（图片检测）
- `test_gui_inputs.go` - GUI界面示例（多输入源）
- `test_input_apis.go` - API使用示例（各种输入源）

## 运行示例

```bash
# 基本示例
go run example/test_simple.go

# GUI示例
go run example/test_gui_inputs.go

# API示例
go run example/test_input_apis.go
```

## 注意事项

- 确保已安装 FFmpeg
- 确保 `yolo12x.onnx` 和 `data.yaml` 文件存在
- 根据你的环境调整 ONNX Runtime 库路径 