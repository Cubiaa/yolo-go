package main

import (
	"fmt"
	"log"
	"yolo-go/gui"
	"yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 摄像头实时检测示例 ===\n")

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
		WithConfThreshold(0.5). // 降低置信度阈值，检测更多对象
		WithIOUThreshold(0.4).
		WithShowFPS(true)

	// 列出可用的摄像头设备
	fmt.Println("📹 可用摄像头设备:")
	cameraDevices := yolo.ListCameraDevices()
	for i, device := range cameraDevices {
		fmt.Printf("   %d. %s\n", i+1, device)
	}

	fmt.Println("\n🎬 启动摄像头检测...")
	fmt.Println("💡 支持的摄像头输入格式:")
	fmt.Println("   - 'camera' 或 'cam' 或 'webcam'")
	fmt.Println("   - '0', '1', '2' (数字索引)")
	fmt.Println("   - 'video=0', 'video=1' (Windows)")
	fmt.Println("   - '/dev/video0', '/dev/video1' (Linux)")

	// 启动摄像头检测
	liveWindow := gui.NewYOLOLiveWindow(detector, "camera", options)
	liveWindow.Run()

	fmt.Println("✅ 摄像头检测完成！")
}
