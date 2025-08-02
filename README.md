# YOLO-Go 实时目标检测系统

🚀 基于Go语言开发的YOLO实时目标检测系统，支持多种输入源和GUI界面

## ✨ 功能特性

- **实时目标检测** - 基于YOLO模型的高精度目标检测
- **多输入源支持** - 视频文件、摄像头、RTSP/RTMP流、屏幕录制
- **GUI界面** - 使用Fyne构建的现代化图形界面
- **GPU加速** - 支持CUDA和DirectML GPU加速

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
    "yolo-go/yolo"
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
    "yolo-go/gui"
    "yolo-go/yolo"
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
    liveWindow := gui.NewYOLOLiveWindow(detector, "test.mp4", options)
    liveWindow.Run()
}
```

### 多输入源示例

```go
// 摄像头检测
liveWindow := gui.NewYOLOLiveWindow(detector, "video=0", options)

// 屏幕录制检测
liveWindow := gui.NewYOLOLiveWindow(detector, "desktop", options)

// RTSP流检测
liveWindow := gui.NewYOLOLiveWindow(detector, "rtsp://192.168.1.100:554/stream", options)

// RTMP流检测
liveWindow := gui.NewYOLOLiveWindow(detector, "rtmp://server.com/live/stream", options)
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
    WithInputSize(640)          // 输入尺寸
```

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
