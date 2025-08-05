# 🖼️ 回调函数中访问逐帧图片指南

本指南详细说明如何在 YOLO-Go 的回调函数中访问和处理逐帧图片数据。

## 📋 概述

在 YOLO-Go 中，所有检测方法的回调函数都使用统一的 `VideoDetectionResult` 结构体，其中包含了每一帧的图片数据：

```go
type VideoDetectionResult struct {
    FrameNumber int           // 帧号
    Timestamp   time.Duration // 时间戳
    Detections  []Detection   // 检测结果
    Image       image.Image   // 当前帧的图片数据 ⭐
}
```

## 🎯 核心功能

### ✅ 可以做什么

1. **访问每一帧的图片数据** - `result.Image` 包含完整的图片信息
2. **保存图片到文件** - 支持 JPEG、PNG 等格式
3. **获取图片属性** - 尺寸、像素数据等
4. **实时图片处理** - 滤镜、裁剪、缩放等
5. **像素级分析** - 访问任意位置的像素值
6. **图片格式转换** - 在不同格式间转换

### 📱 支持的输入源

- 📸 **单张图片** - `Detect(imagePath, options, callback)`
- 🎬 **视频文件** - `Detect(videoPath, options, callback)`
- 📹 **摄像头** - `DetectFromCamera(device, options, callback)`
- 🌐 **RTSP流** - `DetectFromRTSP(rtspURL, options, callback)`
- 📺 **RTMP流** - `DetectFromRTMP(rtmpURL, options, callback)`
- 🖥️ **屏幕录制** - `DetectFromScreen(options, callback)`

## 💡 使用示例

### 1. 基础图片访问

```go
detector.Detect("image.jpg", nil, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        // 获取图片尺寸
        bounds := result.Image.Bounds()
        width, height := bounds.Dx(), bounds.Dy()
        fmt.Printf("图片尺寸: %dx%d\n", width, height)
        
        // 访问像素
        color := result.Image.At(100, 100)
        fmt.Printf("位置(100,100)的颜色: %v\n", color)
    }
})
```

### 2. 保存逐帧图片

```go
detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        // 生成文件名
        filename := fmt.Sprintf("frame_%06d.jpg", result.FrameNumber)
        
        // 保存图片
        file, _ := os.Create(filename)
        defer file.Close()
        jpeg.Encode(file, result.Image, &jpeg.Options{Quality: 90})
        
        fmt.Printf("已保存: %s\n", filename)
    }
})
```

### 3. 视频逐帧处理

```go
detector.Detect("video.mp4", nil, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        fmt.Printf("处理帧 %d (时间: %.3fs)\n", 
            result.FrameNumber, result.Timestamp.Seconds())
        
        // 每10帧保存一次
        if result.FrameNumber%10 == 0 {
            filename := fmt.Sprintf("frame_%d.jpg", result.FrameNumber)
            // 保存逻辑...
        }
        
        // 显示检测结果
        for _, detection := range result.Detections {
            fmt.Printf("检测到: %s (%.1f%%)\n", 
                detection.Class, detection.Score*100)
        }
    }
})
```

### 4. 实时图片分析

```go
detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        // 实时统计
        bounds := result.Image.Bounds()
        pixelCount := bounds.Dx() * bounds.Dy()
        
        fmt.Printf("帧 %d: %dx%d (%d像素), 检测到 %d 个对象\n",
            result.FrameNumber, bounds.Dx(), bounds.Dy(), 
            pixelCount, len(result.Detections))
        
        // 条件保存 - 只保存有检测结果的帧
        if len(result.Detections) > 0 {
            filename := fmt.Sprintf("detection_%d.jpg", result.FrameNumber)
            // 保存逻辑...
        }
    }
})
```

## 🔧 高级图片处理

### 图片格式转换

```go
// 保存为PNG格式
func saveToPNG(img image.Image, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    return png.Encode(file, img)
}

// 保存为JPEG格式
func saveToJPEG(img image.Image, filename string, quality int) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    return jpeg.Encode(file, img, &jpeg.Options{Quality: quality})
}
```

### 图片处理操作

```go
// 使用 github.com/disintegration/imaging 库
import "github.com/disintegration/imaging"

detector.Detect(inputPath, nil, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        // 缩放图片
        resized := imaging.Resize(result.Image, 320, 240, imaging.Lanczos)
        
        // 裁剪图片
        cropped := imaging.Crop(result.Image, image.Rect(0, 0, 300, 300))
        
        // 应用滤镜
        blurred := imaging.Blur(result.Image, 2.0)
        
        // 保存处理后的图片
        imaging.Save(resized, "resized.jpg")
        imaging.Save(cropped, "cropped.jpg")
        imaging.Save(blurred, "blurred.jpg")
    }
})
```

## 📊 性能优化建议

### 1. 选择性保存
```go
// 只保存有检测结果的帧
if len(result.Detections) > 0 {
    // 保存图片
}

// 按间隔保存
if result.FrameNumber%10 == 0 {
    // 每10帧保存一次
}
```

### 2. 异步处理
```go
var imageQueue = make(chan image.Image, 100)

// 在回调中发送到队列
detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
    if result.Image != nil {
        select {
        case imageQueue <- result.Image:
            // 成功发送
        default:
            // 队列满，跳过这一帧
        }
    }
})

// 在另一个goroutine中处理
go func() {
    for img := range imageQueue {
        // 处理图片
        processImage(img)
    }
}()
```

### 3. 内存管理
```go
// 及时释放大图片资源
func processLargeImage(img image.Image) {
    // 处理完成后，让GC回收
    img = nil
    runtime.GC()
}
```

## 🚀 实际应用场景

### 1. 安防监控
```go
// 检测到人员时保存图片
if containsPerson(result.Detections) {
    timestamp := time.Now().Format("2006-01-02_15-04-05")
    filename := fmt.Sprintf("alert_%s.jpg", timestamp)
    saveImage(result.Image, filename)
    sendAlert(filename) // 发送报警
}
```

### 2. 质量检测
```go
// 检测到缺陷时保存图片
if hasDefect(result.Detections) {
    defectDir := "defects"
    filename := fmt.Sprintf("%s/defect_%d.jpg", defectDir, result.FrameNumber)
    saveImage(result.Image, filename)
    logDefect(result.Detections)
}
```

### 3. 数据收集
```go
// 定期保存样本数据
if result.FrameNumber%100 == 0 { // 每100帧
    sampleDir := "samples"
    filename := fmt.Sprintf("%s/sample_%d.jpg", sampleDir, result.FrameNumber)
    saveImage(result.Image, filename)
    saveDetectionData(result.Detections, filename)
}
```

## 📝 注意事项

1. **内存使用** - 图片数据占用较多内存，及时处理和释放
2. **存储空间** - 连续保存图片会占用大量磁盘空间
3. **处理速度** - 复杂的图片处理可能影响实时性能
4. **文件格式** - 选择合适的图片格式平衡质量和大小
5. **并发安全** - 多线程访问时注意线程安全

## 🔗 相关示例文件

- `test_image_access.go` - 完整的图片访问演示
- `test_frame_images.go` - 逐帧图片保存示例
- `test_callback.go` - 基础回调函数使用

## 💬 总结

通过 `VideoDetectionResult.Image` 字段，你可以：
- ✅ 访问每一帧的完整图片数据
- ✅ 保存图片到各种格式
- ✅ 进行实时图片处理和分析
- ✅ 实现自定义的图片处理流水线
- ✅ 构建基于图片的应用逻辑

这为构建复杂的计算机视觉应用提供了强大的基础！