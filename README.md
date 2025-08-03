# YOLO-Go 实时目标检测系统

🚀 基于Go语言开发的YOLO实时目标检测系统，支持多种输入源和GUI界面

## ✨ 功能特性

- **实时目标检测** - 基于YOLO模型的高精度目标检测
- **多输入源支持** - 视频文件、摄像头、RTSP/RTMP流、屏幕录制
- **GUI界面** - 使用Fyne构建的现代化图形界面
- **GPU加速** - 支持CUDA和DirectML GPU加速
- **智能性能优化** - 自动检测硬件并优化CPU/GPU性能
- **多线程并行处理** - 根据CPU核心数动态调整线程数

## 🛠️ 环境要求

- **Go 1.19+** - 编程语言环境
- **FFmpeg** - 视频处理库
- **ONNX Runtime** - 模型推理引擎

## 📦 快速安装

### 1. 安装FFmpeg
```bash
# Windows
winget install FFmpeg

# 或手动下载: https://ffmpeg.org/download.html
```

### 2. 克隆项目
```bash
git clone https://github.com/Cubiaa/yolo-go.git
cd yolo-go
go mod tidy
```

## 🚀 快速开始

### 基本使用

#### 检测图片
```go
package main

import (
    "fmt"
    "log"
    "github.com/Cubiaa/yolo-go/yolo"
)

func main() {
    // 创建检测器
    detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", 
        yolo.DefaultConfig().WithGPU(true))
    if err != nil {
        log.Fatal(err)
    }
    defer detector.Close()

    // 检测图片
    results, err := detector.Detect("cat.jpg", 
        yolo.DefaultDetectionOptions().
            WithDrawBoxes(true).
            WithDrawLabels(true).
            WithConfThreshold(0.9))
    
    if err != nil {
        log.Fatal(err)
    }

    // 保存结果
    results.Save("output.jpg")
    fmt.Printf("检测完成！发现 %d 个对象\n", len(results.Detections))
}
```

#### 启动GUI界面
```go
package main

import (
    "github.com/Cubiaa/yolo-go/gui"
    "github.com/Cubiaa/yolo-go/yolo"
)

func main() {
    // 创建检测器
    detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", 
        yolo.DefaultConfig().WithGPU(true))
    if err != nil {
        log.Fatal(err)
    }
    defer detector.Close()

    // 创建检测选项
    options := yolo.DefaultDetectionOptions().
        WithDrawBoxes(true).
        WithDrawLabels(true).
        WithConfThreshold(0.9).
        WithShowFPS(true)

    // 启动GUI窗口
    liveWindow := gui.NewYOLOLiveWindow(detector, gui.InputTypeFile, "test.mp4", options)
    liveWindow.Run()
}
```

### 多输入源示例

```go
// 摄像头检测
liveWindow := gui.NewYOLOLiveWindow(detector, gui.InputTypeCamera, "0", options)

// 屏幕录制检测
liveWindow := gui.NewYOLOLiveWindow(detector, gui.InputTypeScreen, "desktop", options)

// RTSP流检测
liveWindow := gui.NewYOLOLiveWindow(detector, gui.InputTypeRTSP, "rtsp://192.168.1.100:554/stream", options)

// RTMP流检测
liveWindow := gui.NewYOLOLiveWindow(detector, gui.InputTypeRTMP, "rtmp://server.com/live/stream", options)
```

## ⚙️ 配置选项

```go
// 检测选项
options := yolo.DefaultDetectionOptions().
    WithDrawBoxes(true).        // 绘制检测框
    WithDrawLabels(true).       // 绘制标签
    WithConfThreshold(0.9).     // 置信度阈值
    WithIOUThreshold(0.4).      // IOU阈值
    WithShowFPS(true).          // 显示FPS

// GPU配置
config := yolo.DefaultConfig().
    WithGPU(true).              // 启用GPU
    WithGPUDeviceID(0).         // 绑定GPU设备ID（默认0）
    WithInputSize(640)          // 输入尺寸

// 🆕 自动检测模型输入尺寸（推荐）
autoConfig := yolo.AutoDetectInputSizeConfig("model.onnx")
// 自动从模型文件名或模型元数据中检测合适的输入尺寸
```

## 🚀 性能优化

### 默认高性能配置（推荐）
```go
// 默认配置现在就是高性能配置，自动检测硬件并优化
detector, err := yolo.NewYOLO("model.onnx", "config.yaml")
// 或者显式使用高性能配置（效果相同）
detector, err := yolo.NewYOLO("model.onnx", "config.yaml", yolo.HighPerformanceConfig())
```

### 手动性能配置
```go
// GPU最大性能配置
gpuConfig := yolo.MaxPerformanceGPUConfig()  // 640x640输入，GPU加速

// CPU最大性能配置  
cpuConfig := yolo.MaxPerformanceCPUConfig()  // 416x416输入，多线程优化

// 检查GPU支持
yolo.CheckGPUSupport()

// 检测GPU是否可用
if yolo.IsGPUAvailable() {
    fmt.Println("GPU可用")
}
```

### 性能优化特性
- **智能硬件检测** - 自动检测GPU支持并选择最优配置
- **动态线程调整** - 根据CPU核心数自动调整线程数
- **图优化** - 启用ONNX Runtime的所有图优化
- **并行执行** - 使用并行执行模式提升性能
- **输入尺寸优化** - GPU使用640x640，CPU使用416x416

### 性能建议
- **CPU模式**: 使用较小输入尺寸(416x416)以提高速度
- **GPU模式**: 使用较大输入尺寸(640x640)以提高精度
- **批量处理**: 处理多个图像时使用批量模式
- **阈值调整**: 适当调整置信度和IOU阈值平衡精度与速度

### 🔥 多GPU支持
```go
// 单GPU绑定特定设备
detector1, _ := yolo.NewYOLO("model.onnx", "config.yaml",
    yolo.DefaultConfig().WithGPU(true).WithGPUDeviceID(0))

// 多GPU并行处理
detector2, _ := yolo.NewYOLO("model.onnx", "config.yaml",
    yolo.DefaultConfig().WithGPU(true).WithGPUDeviceID(1))

// 并行检测不同输入
go detector1.Detect("image1.jpg")
go detector2.Detect("image2.jpg")
```

**多GPU特性：**
- ✅ **设备绑定**：支持绑定特定GPU设备ID
- ✅ **并行处理**：多检测器实例同时工作
- ✅ **负载均衡**：轮询分配GPU资源
- ✅ **批量优化**：适合大批量图像处理

## 🔧 智能模型适配

### 智能模型适配
```go
// 推荐：使用默认配置（已集成智能模型适配）
detector, err := yolo.NewYOLO("your-model.onnx", "config.yaml", 
    yolo.DefaultConfig().WithGPU(true))

// 或者分步配置
config := yolo.DefaultConfig()
detector, err := yolo.NewYOLO("your-model.onnx", "config.yaml", config)
```

####### 智能适配特性

- ✅ **自动硬件检测**：智能选择GPU/CPU最优配置
- ✅ **模型尺寸适配**：运行时自动检测并调整输入尺寸
- ✅ **链式配置**：支持 `.WithGPU()` `.WithLibraryPath()` 等方法
- ✅ **向下兼容**：保持原有API使用方式不变

### 支持的检测方式
1. **文件名检测**: 从模型文件名中提取尺寸信息
   - `yolo11n-640.onnx` → 640x640
   - `custom-model-416.onnx` → 416x416
   - `yolo8s-1280.onnx` → 1280x1280

2. **模型类型推断**: 根据YOLO版本推断标准尺寸
   - YOLOv8/v11 系列 → 640x640
   - YOLOv12 系列 → 640x640
   - 自定义模型 → 640x640 (默认)

3. **实时模型信息**: 运行时从ONNX模型中读取实际输入输出形状
   - 自动验证配置与模型的匹配性
   - 动态调整张量形状
   - 智能错误提示和建议

### 模型兼容性
- ✅ **YOLOv8**: 完全支持 (n/s/m/l/x)
- ✅ **YOLOv11**: 完全支持 (n/s/m/l/x) 
- ✅ **YOLOv12**: 完全支持
- ✅ **自定义YOLO**: 支持标准ONNX格式
- ✅ **多尺寸**: 320/416/512/640/736/832/1024/1280

### 使用建议
```go
// 🎯 最佳实践：让系统自动处理一切
detector, err := yolo.NewYOLO("model.onnx", "config.yaml", 
    yolo.DefaultConfig().WithGPU(true))

// 🔧 手动指定（当自动检测不准确时）
detector, err := yolo.NewYOLO("model.onnx", "config.yaml",
    yolo.ExtremePerformanceConfig().WithInputSize(1280))
```

#### 使用建议
- **推荐方式**：直接使用 `DefaultConfig()` 即可获得最优性能和自动适配
- **链式调用**：`yolo.DefaultConfig().WithGPU(true).WithLibraryPath("path")`
- **自动回退**：检测失败时自动使用默认配置（640x640）
- **零配置**：支持运行时动态调整，无需手动设置

## 🔧 常见问题

### FFmpeg未安装
```
错误: vidio: ffmpeg is not installed
解决: 安装FFmpeg并确保在PATH中
```

### GPU初始化失败
```
错误: GPU initialization panic
解决: 检查CUDA/DirectML安装，或使用CPU模式
```

## 🙏 致谢

- [ONNX Runtime](https://onnxruntime.ai/) - 模型推理引擎
- [Fyne](https://fyne.io/) - GUI框架
- [FFmpeg](https://ffmpeg.org/) - 视频处理
- [Vidio](https://github.com/AlexEidt/Vidio) - Go视频处理库

---

⭐ 如果这个项目对你有帮助，请给个Star！
