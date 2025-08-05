package main

import (
	"fmt"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🎯 YOLO-Go 统一回调函数示例")
	fmt.Println("========================================")

	// 初始化YOLO检测器
	detector, err := yolo.NewYOLO("models/yolo11n.onnx", "coco.yaml")
	if err != nil {
		fmt.Printf("❌ 初始化YOLO失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 设置检测选项
	options := &yolo.DetectionOptions{
		ConfidenceThreshold: 0.5,
		IOUThreshold:        0.4,
		MaxDetections:       100,
	}

	// 统一的回调函数，适用于所有检测方法
	unifiedCallback := func(result yolo.VideoDetectionResult) {
		fmt.Printf("📊 帧 %d (%.2fs): 检测到 %d 个对象\n", 
			result.FrameNumber, result.Timestamp.Seconds(), len(result.Detections))
		
		// 输出检测结果详情
		for i, detection := range result.Detections {
			fmt.Printf("  对象 %d: %s (置信度: %.2f%%)\n",
				i+1, detection.Class, detection.Score*100)
		}
	}

	fmt.Println("\n🎥 测试视频文件检测...")
	testVideoDetection(detector, options, unifiedCallback)

	fmt.Println("\n📹 测试摄像头检测...")
	testCameraDetection(detector, options, unifiedCallback)

	fmt.Println("\n📺 测试RTSP流检测...")
	testRTSPDetection(detector, options, unifiedCallback)

	fmt.Println("\n🖥️ 测试屏幕检测...")
	testScreenDetection(detector, options, unifiedCallback)
}

// 测试视频文件检测
func testVideoDetection(detector *yolo.YOLO, options *yolo.DetectionOptions, callback func(yolo.VideoDetectionResult)) {
	// 使用统一的Detect API
	results, err := detector.Detect("test_video.mp4", options, callback)
	if err != nil {
		fmt.Printf("❌ 视频检测失败: %v\n", err)
		return
	}
	fmt.Printf("✅ 视频检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 测试摄像头检测
func testCameraDetection(detector *yolo.YOLO, options *yolo.DetectionOptions, callback func(yolo.VideoDetectionResult)) {
	// 设置超时，避免无限运行
	go func() {
		time.Sleep(10 * time.Second)
		fmt.Println("⏰ 摄像头检测超时，停止检测")
	}()

	results, err := detector.DetectFromCamera("0", options, callback)
	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
		return
	}
	fmt.Printf("✅ 摄像头检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 测试RTSP流检测
func testRTSPDetection(detector *yolo.YOLO, options *yolo.DetectionOptions, callback func(yolo.VideoDetectionResult)) {
	// 设置超时，避免无限运行
	go func() {
		time.Sleep(10 * time.Second)
		fmt.Println("⏰ RTSP检测超时，停止检测")
	}()

	results, err := detector.DetectFromRTSP("rtsp://example.com/stream", options, callback)
	if err != nil {
		fmt.Printf("❌ RTSP检测失败: %v\n", err)
		return
	}
	fmt.Printf("✅ RTSP检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}

// 测试屏幕检测
func testScreenDetection(detector *yolo.YOLO, options *yolo.DetectionOptions, callback func(yolo.VideoDetectionResult)) {
	// 设置超时，避免无限运行
	go func() {
		time.Sleep(10 * time.Second)
		fmt.Println("⏰ 屏幕检测超时，停止检测")
	}()

	results, err := detector.DetectFromScreen(options, callback)
	if err != nil {
		fmt.Printf("❌ 屏幕检测失败: %v\n", err)
		return
	}
	fmt.Printf("✅ 屏幕检测完成！总共检测到 %d 个对象\n", len(results.Detections))
}