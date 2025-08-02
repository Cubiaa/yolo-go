package main

import (
	"fmt"
	"log"
	"yolo-go/gui"
	"yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO GUI 多输入源测试 ===\n")

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

	// 测试1：从视频文件启动GUI
	fmt.Println("🎬 测试1：从视频文件启动GUI")
	liveWindow1 := gui.NewYOLOLiveWindow(detector, "test.mp4", options)
	liveWindow1.Run()

	// 测试2：从摄像头启动GUI
	fmt.Println("🎬 测试2：从摄像头启动GUI")
	fmt.Println("📹 可用摄像头设备:")
	cameraDevices := yolo.ListCameraDevices()
	for i, device := range cameraDevices {
		fmt.Printf("   %d. %s\n", i+1, device)
	}

	// 使用默认摄像头
	liveWindow2 := gui.NewYOLOLiveWindow(detector, "video=0", options)
	liveWindow2.Run()

	// 测试3：从RTSP流启动GUI
	fmt.Println("🎬 测试3：从RTSP流启动GUI")
	liveWindow3 := gui.NewYOLOLiveWindow(detector, "rtsp://192.168.1.100:554/stream", options)
	liveWindow3.Run()

	// 测试4：从屏幕录制启动GUI
	fmt.Println("🎬 测试4：从屏幕录制启动GUI")
	fmt.Println("🖥️  可用屏幕设备:")
	screenDevices := yolo.ListScreenDevices()
	for i, device := range screenDevices {
		fmt.Printf("   %d. %s\n", i+1, device)
	}

	// 使用主屏幕
	liveWindow4 := gui.NewYOLOLiveWindow(detector, "desktop", options)
	liveWindow4.Run()

	// 测试5：从RTMP流启动GUI
	fmt.Println("🎬 测试5：从RTMP流启动GUI")
	liveWindow5 := gui.NewYOLOLiveWindow(detector, "rtmp://server.com/live/stream", options)
	liveWindow5.Run()

	fmt.Println("✅ 多输入源GUI测试完成！")
}
