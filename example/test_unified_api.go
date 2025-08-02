package main

import (
	"fmt"
	"log"

	"github.com/Cubiaa/yolo-go/gui"
	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 统一 API 使用示例 ===\n")

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

	fmt.Println("💡 使用统一的 NewYOLOLiveWindow API:")
	fmt.Println("   参数1: detector - YOLO检测器")
	fmt.Println("   参数2: inputType - 输入源类型 (使用常量)")
	fmt.Println("   参数3: inputPath - 输入路径")
	fmt.Println("   参数4: options - 检测选项")

	// 示例1：摄像头检测
	fmt.Println("\n🎬 示例1：摄像头检测")
	fmt.Println("   inputType: gui.InputTypeCamera")
	fmt.Println("   inputPath: '0' (第一个摄像头)")
	liveWindow1 := gui.NewYOLOLiveWindow(detector, gui.InputTypeCamera, "0", options)
	liveWindow1.Run()

	// 示例2：视频文件检测
	fmt.Println("\n🎬 示例2：视频文件检测")
	fmt.Println("   inputType: gui.InputTypeFile")
	fmt.Println("   inputPath: 'test.mp4'")
	liveWindow2 := gui.NewYOLOLiveWindow(detector, gui.InputTypeFile, "test.mp4", options)
	liveWindow2.Run()

	// 示例3：RTSP流检测
	fmt.Println("\n🎬 示例3：RTSP流检测")
	fmt.Println("   inputType: gui.InputTypeRTSP")
	fmt.Println("   inputPath: 'rtsp://192.168.1.100:554/stream'")
	liveWindow3 := gui.NewYOLOLiveWindow(detector, gui.InputTypeRTSP, "rtsp://192.168.1.100:554/stream", options)
	liveWindow3.Run()

	// 示例4：屏幕录制检测
	fmt.Println("\n🎬 示例4：屏幕录制检测")
	fmt.Println("   inputType: gui.InputTypeScreen")
	fmt.Println("   inputPath: 'desktop'")
	liveWindow4 := gui.NewYOLOLiveWindow(detector, gui.InputTypeScreen, "desktop", options)
	liveWindow4.Run()

	// 示例5：RTMP流检测
	fmt.Println("\n🎬 示例5：RTMP流检测")
	fmt.Println("   inputType: gui.InputTypeRTMP")
	fmt.Println("   inputPath: 'rtmp://server.com/live/stream'")
	liveWindow5 := gui.NewYOLOLiveWindow(detector, gui.InputTypeRTMP, "rtmp://server.com/live/stream", options)
	liveWindow5.Run()

	fmt.Println("\n✅ 统一 API 示例完成！")
	fmt.Println("💡 优势:")
	fmt.Println("   - 只有一个 API，避免混淆")
	fmt.Println("   - 明确指定类型，避免歧义")
	fmt.Println("   - 使用常量，类型安全")
	fmt.Println("   - 代码清晰，意图明确")
}
