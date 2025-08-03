package main

import (
	"fmt"
	"log"

	"github.com/Cubiaa/yolo/gui"
	"github.com/Cubiaa/yolo/yolo"
)

func main() {
	fmt.Println("=== YOLO 摄像头设备检测示例 ===\n")

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
		WithConfThreshold(0.5).
		WithIOUThreshold(0.4).
		WithShowFPS(true)

	// 列出所有可用的摄像头设备
	fmt.Println("📹 可用摄像头设备:")
	cameraDevices := yolo.ListCameraDevices()
	for i, device := range cameraDevices {
		fmt.Printf("   %d. %s\n", i+1, device)
	}

	// 获取摄像头设备信息
	fmt.Println("\n🔍 摄像头设备详细信息:")
	deviceInfo := yolo.GetCameraDeviceInfo()
	for key, device := range deviceInfo {
		fmt.Printf("   %s: %s\n", key, device)
	}

	fmt.Println("\n💡 摄像头使用说明:")
	fmt.Println("1. 使用 'camera' 关键字 - 自动选择默认摄像头")
	fmt.Println("2. 使用数字索引 - 选择特定摄像头 (0, 1, 2...)")
	fmt.Println("3. 使用设备路径 - 直接指定设备 (video=0, /dev/video0)")

	// 示例1：使用默认摄像头
	fmt.Println("\n🎬 示例1：使用默认摄像头 ('camera')")
	liveWindow1 := gui.NewYOLOLiveWindow(detector, "camera", options)
	liveWindow1.Run()

	// 示例2：使用第一个摄像头
	fmt.Println("\n🎬 示例2：使用第一个摄像头 ('0')")
	liveWindow2 := gui.NewYOLOLiveWindow(detector, "0", options)
	liveWindow2.Run()

	// 示例3：使用第二个摄像头
	fmt.Println("\n🎬 示例3：使用第二个摄像头 ('1')")
	liveWindow3 := gui.NewYOLOLiveWindow(detector, "1", options)
	liveWindow3.Run()

	// 示例4：使用Windows设备路径
	fmt.Println("\n🎬 示例4：使用Windows设备路径 ('video=0')")
	liveWindow4 := gui.NewYOLOLiveWindow(detector, "video=0", options)
	liveWindow4.Run()

	fmt.Println("✅ 摄像头设备检测示例完成！")
}
