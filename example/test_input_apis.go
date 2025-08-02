package main

import (
	"fmt"
	"log"
	"strings"
	"yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 多输入源检测测试 ===\n")

	// 创建检测器
	LibPath := "D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll"
	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml",
		yolo.DefaultConfig().WithGPU(true).WithLibraryPath(LibPath))
	if err != nil {
		log.Fatalf("创建检测器失败: %v", err)
	}
	defer detector.Close()

	// 创建检测选项
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithDrawLabels(true).
		WithConfThreshold(0.9).
		WithIOUThreshold(0.4).
		WithShowFPS(true)

	// 测试1：从摄像头检测
	fmt.Println("🎬 测试1：从摄像头检测")
	results1, err := detector.DetectFromCamera("0", options) // 使用默认摄像头
	if err != nil {
		fmt.Printf("❌ 摄像头检测失败: %v\n", err)
	} else {
		fmt.Printf("✅ 摄像头检测完成！检测到 %d 个对象\n", len(results1.Detections))
	}

	fmt.Println("\n" + strings.Repeat("=", 50) + "\n")

	// 测试2：从RTSP流检测
	fmt.Println("🎬 测试2：从RTSP流检测")
	results2, err := detector.DetectFromRTSP("rtsp://192.168.1.100:554/stream", options)
	if err != nil {
		fmt.Printf("❌ RTSP检测失败: %v\n", err)
	} else {
		fmt.Printf("✅ RTSP检测完成！检测到 %d 个对象\n", len(results2.Detections))
	}

	fmt.Println("\n" + strings.Repeat("=", 50) + "\n")

	// 测试3：从屏幕录制检测
	fmt.Println("🎬 测试3：从屏幕录制检测")
	results3, err := detector.DetectFromScreen(options)
	if err != nil {
		fmt.Printf("❌ 屏幕检测失败: %v\n", err)
	} else {
		fmt.Printf("✅ 屏幕检测完成！检测到 %d 个对象\n", len(results3.Detections))
	}

	fmt.Println("\n" + strings.Repeat("=", 50) + "\n")

	// 测试4：从RTMP流检测
	fmt.Println("🎬 测试4：从RTMP流检测")
	results4, err := detector.DetectFromRTMP("rtmp://server.com/live/stream", options)
	if err != nil {
		fmt.Printf("❌ RTMP检测失败: %v\n", err)
	} else {
		fmt.Printf("✅ RTMP检测完成！检测到 %d 个对象\n", len(results4.Detections))
	}

	fmt.Println("\n🎯 多输入源检测测试完成！")
}
