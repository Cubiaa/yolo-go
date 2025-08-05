package main

import (
	"fmt"
	"image"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🧪 测试YOLO回调函数功能")

	// 创建YOLO检测器配置
	config := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputDimensions(640, 640)

	// 初始化YOLO检测器
	detector, err := yolo.NewYOLO("yolov8n.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 设置检测选项
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
	}

	// 测试1: 图片检测回调函数
	fmt.Println("\n📸 测试图片检测回调函数")
	testImageCallback(detector)

	// 测试2: 视频检测回调函数
	fmt.Println("\n🎬 测试视频检测回调函数")
	testVideoCallback(detector, options)

	// 测试3: 摄像头检测回调函数
	fmt.Println("\n📹 测试摄像头检测回调函数")
	testCameraCallback(detector, options)

	// 测试4: RTSP流检测回调函数
	fmt.Println("\n🌐 测试RTSP流检测回调函数")
	testRTSPCallback(detector, options)

	// 测试5: RTMP流检测回调函数
	fmt.Println("\n📡 测试RTMP流检测回调函数")
	testRTMPCallback(detector, options)

	// 测试6: 屏幕录制检测回调函数
	fmt.Println("\n🖥️  测试屏幕录制检测回调函数")
	testScreenCallback(detector, options)

	fmt.Println("\n✅ 所有回调函数测试完成！")
}

// 测试图片检测回调函数
func testImageCallback(detector *yolo.YOLO) {
	imagePath := "test_image.jpg" // 请确保有测试图片

	// 使用统一的Detect API
	detector.Detect(imagePath, nil, func(detections []yolo.Detection, err error) {
		if err != nil {
			fmt.Printf("❌ 图片检测失败: %v\n", err)
			return
		}

		fmt.Printf("📊 图片检测结果: 发现 %d 个对象\n", len(detections))
		for i, detection := range detections {
			fmt.Printf("  对象 %d: %s (置信度: %.2f%%, 坐标: [%.1f, %.1f, %.1f, %.1f])\n",
				i+1, detection.Class, detection.Score*100,
				detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
		}
	})
}

// 测试视频检测回调函数
func testVideoCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	videoPath := "test_video.mp4" // 请确保有测试视频

	var frameCount int
	var totalDetections int
	startTime := time.Now()

	// 使用统一的Detect API
	_, err := detector.Detect(videoPath, options, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalDetections += len(result.Detections)

		// 每10帧输出一次统计信息
		if frameCount%10 == 0 {
			elapsed := time.Since(startTime)
			fps := float64(frameCount) / elapsed.Seconds()
			fmt.Printf("📊 帧 %d: 检测到 %d 个对象, FPS: %.1f, 时间戳: %v\n",
				result.FrameNumber, len(result.Detections), fps, result.Timestamp)
		}

		// 输出检测到的对象详情（仅前3帧）
		if frameCount <= 3 {
			for _, detection := range result.Detections {
				fmt.Printf("  -> %s (%.2f%%)\n", detection.Class, detection.Score*100)
			}
		}
	})

	if err != nil {
		fmt.Printf("❌ 视频检测失败: %v\n", err)
	} else {
		elapsed := time.Since(startTime)
		avgFPS := float64(frameCount) / elapsed.Seconds()
		fmt.Printf("✅ 视频处理完成: %d 帧, 总检测数: %d, 平均FPS: %.1f\n",
			frameCount, totalDetections, avgFPS)
	}
}

// 测试摄像头检测回调函数
func testCameraCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	device := "0" // 默认摄像头

	var frameCount int
	startTime := time.Now()
	maxFrames := 50 // 限制处理帧数以避免无限运行

	// 使用统一的DetectFromCamera API
	_, err := detector.DetectFromCamera(device, options, func(img image.Image, detections []yolo.Detection, err error) {
		if err != nil {
			fmt.Printf("❌ 摄像头检测错误: %v\n", err)
			return
		}

		frameCount++

		// 每5帧输出一次统计信息
		if frameCount%5 == 0 {
			elapsed := time.Since(startTime)
			fps := float64(frameCount) / elapsed.Seconds()
			fmt.Printf("📹 摄像头帧 %d: 检测到 %d 个对象, FPS: %.1f\n",
				frameCount, len(detections), fps)

			// 输出检测到的对象
			for _, detection := range detections {
				fmt.Printf("  -> %s (%.2f)\n", detection.Class, detection.Score)
			}
		}

		// 达到最大帧数后停止
		if frameCount >= maxFrames {
			fmt.Printf("🛑 达到最大帧数限制 (%d 帧)，停止摄像头检测\n", maxFrames)
			return
		}
	})

	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
	} else {
		fmt.Printf("✅ 摄像头检测完成: 处理了 %d 帧\n", frameCount)
	}
}

// 测试RTSP流检测回调函数
func testRTSPCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	// RTSP流地址示例 (请替换为实际的RTSP地址)
	rtspURL := "rtsp://example.com:554/stream"
	// 或者使用本地测试RTSP流: "rtsp://127.0.0.1:8554/test"

	fmt.Printf("🔗 连接RTSP流: %s\n", rtspURL)

	frameCount := 0
	maxFrames := 50 // 限制处理帧数
	startTime := time.Now()

	_, err := detector.DetectFromRTSP(rtspURL, options, func(result yolo.VideoDetectionResult) {
		frameCount++
		elapsed := time.Since(startTime)
		fps := float64(frameCount) / elapsed.Seconds()

		fmt.Printf("📺 RTSP帧 %d - 检测到 %d 个对象 - FPS: %.2f\n",
			result.FrameNumber, len(result.Detections), fps)

		// 显示检测详情
		for i, detection := range result.Detections {
			if i < 3 { // 只显示前3个检测结果
				fmt.Printf("  🎯 %s (%.2f%%) [%.0f,%.0f,%.0f,%.0f]\n", 
					detection.Class, detection.Score*100,
					detection.Box[0], detection.Box[1],
					detection.Box[2], detection.Box[3])
			}
		}

		// 达到最大帧数时停止
		if frameCount >= maxFrames {
			fmt.Printf("⏹️  已处理 %d 帧，停止RTSP检测\n", maxFrames)
			return
		}
	})

	if err != nil {
		fmt.Printf("❌ RTSP流检测失败: %v\n", err)
		fmt.Println("💡 提示: 请确保RTSP地址正确且可访问")
	} else {
		fmt.Printf("✅ RTSP检测完成，共处理 %d 帧\n", frameCount)
	}
}

// 测试RTMP流检测回调函数
func testRTMPCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	// RTMP流地址示例 (请替换为实际的RTMP地址)
	rtmpURL := "rtmp://example.com/live/stream"
	// 或者使用本地测试RTMP流: "rtmp://127.0.0.1:1935/live/test"

	fmt.Printf("🔗 连接RTMP流: %s\n", rtmpURL)

	frameCount := 0
	maxFrames := 50 // 限制处理帧数
	startTime := time.Now()

	_, err := detector.DetectFromRTMP(rtmpURL, options, func(result yolo.VideoDetectionResult) {
		frameCount++
		elapsed := time.Since(startTime)
		fps := float64(frameCount) / elapsed.Seconds()

		fmt.Printf("📡 RTMP帧 %d - 检测到 %d 个对象 - FPS: %.2f\n",
			result.FrameNumber, len(result.Detections), fps)

		// 显示检测详情
		for i, detection := range result.Detections {
			if i < 3 { // 只显示前3个检测结果
				fmt.Printf("  🎯 %s (%.2f%%) [%.0f,%.0f,%.0f,%.0f]\n", 
					detection.Class, detection.Score*100,
					detection.Box[0], detection.Box[1],
					detection.Box[2], detection.Box[3])
			}
		}

		// 达到最大帧数时停止
		if frameCount >= maxFrames {
			fmt.Printf("⏹️  已处理 %d 帧，停止RTMP检测\n", maxFrames)
			return
		}
	})

	if err != nil {
		fmt.Printf("❌ RTMP流检测失败: %v\n", err)
		fmt.Println("💡 提示: 请确保RTMP地址正确且可访问")
	} else {
		fmt.Printf("✅ RTMP检测完成，共处理 %d 帧\n", frameCount)
	}
}

// 测试屏幕录制检测回调函数
func testScreenCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("🖥️  开始屏幕录制检测...")

	frameCount := 0
	maxFrames := 30 // 限制处理帧数，避免长时间运行
	startTime := time.Now()

	_, err := detector.DetectFromScreen(options, func(result yolo.VideoDetectionResult) {
		frameCount++
		elapsed := time.Since(startTime)
		fps := float64(frameCount) / elapsed.Seconds()

		fmt.Printf("🖥️  屏幕帧 %d - 检测到 %d 个对象 - FPS: %.2f\n",
			result.FrameNumber, len(result.Detections), fps)

		// 显示检测详情
		for i, detection := range result.Detections {
			if i < 5 { // 显示前5个检测结果
				fmt.Printf("  🎯 %s (%.2f%%) [%.0f,%.0f,%.0f,%.0f]\n", 
					detection.Class, detection.Score*100,
					detection.Box[0], detection.Box[1],
					detection.Box[2], detection.Box[3])
			}
		}

		// 达到最大帧数时停止
		if frameCount >= maxFrames {
			fmt.Printf("⏹️  已处理 %d 帧，停止屏幕检测\n", maxFrames)
			return
		}
	})

	if err != nil {
		fmt.Printf("❌ 屏幕录制检测失败: %v\n", err)
		fmt.Println("💡 提示: 请确保有屏幕录制权限")
	} else {
		fmt.Printf("✅ 屏幕检测完成，共处理 %d 帧\n", frameCount)
	}
}

// 额外示例：自定义回调函数处理逻辑
func customVideoCallback(result yolo.VideoDetectionResult) {
	// 自定义处理逻辑示例
	if len(result.Detections) > 0 {
		// 统计各类别数量
		classCount := make(map[string]int)
		for _, detection := range result.Detections {
			classCount[detection.Class]++
		}

		// 输出统计结果
		fmt.Printf("帧 %d 检测统计: ", result.FrameNumber)
		for class, count := range classCount {
			fmt.Printf("%s:%d ", class, count)
		}
		fmt.Println()

		// 可以在这里添加更多自定义逻辑：
		// - 保存特定帧到文件
		// - 发送检测结果到数据库
		// - 触发报警机制
		// - 实时数据分析
	}
}
