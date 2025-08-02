package main

import (
	"fmt"
	"log"
	"yolo-go/gui"
	"yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 实时检测测试 ===\n")

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
		WithShowFPS(true).
		WithLabelColor("red").
		WithBoxColor("blue")

	// 直接启动GUI窗口进行实时检测
	fmt.Println("🎬 启动实时检测窗口...")
	liveWindow := gui.NewYOLOLiveWindow(detector, "test.mp4", options)
	liveWindow.Run()

	fmt.Println("✅ 实时检测完成！")
}
