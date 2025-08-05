package main

import (
	"fmt"
	"image"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🧪 测试可选回调函数功能")

	// 创建YOLO检测器配置
	config := yolo.DefaultConfig().WithGPU(false)

	// 创建YOLO检测器
	detector, err := yolo.NewYOLO("yolov8n.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 创建检测选项
	options := &yolo.DetectionOptions{
		ConfThreshold: 0.5,
		IOUThreshold:  0.4,
	}

	fmt.Println("\n=== 演示可选回调函数的使用方式 ===")

	// 方式1：不使用回调函数（传统方式）
	fmt.Println("\n📹 方式1：摄像头检测 - 不使用回调函数")
	testCameraWithoutCallback(detector, options)

	// 方式2：使用回调函数
	fmt.Println("\n📹 方式2：摄像头检测 - 使用回调函数")
	testCameraWithCallback(detector, options)

	// 方式3：RTSP流检测 - 不使用回调函数
	fmt.Println("\n🌐 方式3：RTSP流检测 - 不使用回调函数")
	testRTSPWithoutCallback(detector, options)

	// 方式4：RTSP流检测 - 使用回调函数
	fmt.Println("\n🌐 方式4：RTSP流检测 - 使用回调函数")
	testRTSPWithCallback(detector, options)

	// 方式5：屏幕录制检测 - 不使用回调函数
	fmt.Println("\n🖥️ 方式5：屏幕录制检测 - 不使用回调函数")
	testScreenWithoutCallback(detector, options)

	// 方式6：屏幕录制检测 - 使用回调函数
	fmt.Println("\n🖥️ 方式6：屏幕录制检测 - 使用回调函数")
	testScreenWithCallback(detector, options)

	fmt.Println("\n✅ 可选回调函数测试完成！")
}

// 摄像头检测 - 不使用回调函数
func testCameraWithoutCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动摄像头检测（传统方式）...")
	
	// 不传递回调函数，使用传统的返回值方式
	results, err := detector.DetectFromCamera("0", options)
	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ 摄像头检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 摄像头检测 - 使用回调函数
func testCameraWithCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动摄像头检测（回调方式）...")
	
	var frameCount int
	var totalDetections int
	
	// 传递回调函数进行实时处理，使用统一的VideoDetectionResult
	results, err := detector.DetectFromCamera("0", options, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalDetections += len(result.Detections)
		
		// 每10帧输出一次统计信息
		if frameCount%10 == 0 {
			fmt.Printf("📊 已处理 %d 帧 (帧号: %d, 时间戳: %.2fs)，平均每帧检测到 %.1f 个对象\n", 
				frameCount, result.FrameNumber, result.Timestamp.Seconds(), float64(totalDetections)/float64(frameCount))
		}
		
		// 输出检测结果详情
		for i, detection := range result.Detections {
			fmt.Printf("  对象 %d: %s (置信度: %.2f%%, 位置: [%.0f,%.0f,%.0f,%.0f])\n",
				i+1, detection.Class, detection.Score*100,
				detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
		}
	})
	
	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ 摄像头检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// RTSP流检测 - 不使用回调函数
func testRTSPWithoutCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动RTSP流检测（传统方式）...")
	
	// 不传递回调函数
	results, err := detector.DetectFromRTSP("rtsp://example.com/stream", options)
	if err != nil {
		fmt.Printf("❌ RTSP检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ RTSP检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// RTSP流检测 - 使用回调函数
func testRTSPWithCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动RTSP流检测（回调方式）...")
	
	var frameCount int
	var totalDetections int
	startTime := time.Now()
	
	// 传递回调函数进行实时处理
	results, err := detector.DetectFromRTSP("rtsp://example.com/stream", options, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalDetections += len(result.Detections)
		
		// 计算FPS
		elapsed := time.Since(startTime).Seconds()
		fps := float64(frameCount) / elapsed
		
		// 每5帧输出一次统计信息
		if frameCount%5 == 0 {
			fmt.Printf("📊 RTSP帧 %d (FPS: %.1f), 检测到 %d 个对象\n", 
				result.FrameNumber, fps, len(result.Detections))
		}
		
		// 输出检测结果详情
		for i, detection := range result.Detections {
			fmt.Printf("  对象 %d: %s (置信度: %.2f%%, 位置: [%.0f,%.0f,%.0f,%.0f])\n",
				i+1, detection.Class, detection.Score*100,
				detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
		}
	})
	
	if err != nil {
		fmt.Printf("❌ RTSP检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ RTSP检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 屏幕录制检测 - 不使用回调函数
func testScreenWithoutCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动屏幕录制检测（传统方式）...")
	
	// 不传递回调函数
	results, err := detector.DetectFromScreen(options)
	if err != nil {
		fmt.Printf("❌ 屏幕检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ 屏幕检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 屏幕录制检测 - 使用回调函数
func testScreenWithCallback(detector *yolo.YOLO, options *yolo.DetectionOptions) {
	fmt.Println("启动屏幕录制检测（回调方式）...")
	
	var frameCount int
	var totalDetections int
	startTime := time.Now()
	
	// 传递回调函数进行实时处理
	results, err := detector.DetectFromScreen(options, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalDetections += len(result.Detections)
		
		// 计算FPS
		elapsed := time.Since(startTime).Seconds()
		fps := float64(frameCount) / elapsed
		
		// 每3帧输出一次统计信息
		if frameCount%3 == 0 {
			fmt.Printf("📊 屏幕帧 %d (FPS: %.1f), 检测到 %d 个对象\n", 
				result.FrameNumber, fps, len(result.Detections))
		}
		
		// 输出检测结果详情
		for i, detection := range result.Detections {
			fmt.Printf("  对象 %d: %s (置信度: %.2f%%, 位置: [%.0f,%.0f,%.0f,%.0f])\n",
				i+1, detection.Class, detection.Score*100,
				detection.Box[0], detection.Box[1], detection.Box[2], detection.Box[3])
		}
	})
	
	if err != nil {
		fmt.Printf("❌ 屏幕检测失败: %v\n", err)
		return
	}
	
	fmt.Printf("✅ 屏幕检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}