package main

import (
	"fmt"
	"log"
	"yolo-go/gui"
	"yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO 明确输入源类型示例 ===\n")

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

	fmt.Println("💡 使用 NewYOLOLiveWindowWithType API 明确指定输入源类型:")
	fmt.Println("   参数1: detector - YOLO检测器")
	fmt.Println("   参数2: inputType - 输入源类型 ('file', 'camera', 'rtsp', 'rtmp', 'screen')")
	fmt.Println("   参数3: inputPath - 输入路径")
	fmt.Println("   参数4: options - 检测选项")

	// 示例1：明确指定为摄像头
	fmt.Println("\n🎬 示例1：明确指定为摄像头")
	fmt.Println("   inputType: gui.InputTypeCamera")
	fmt.Println("   inputPath: '0' (第一个摄像头)")
	liveWindow1 := gui.NewYOLOLiveWindowWithType(detector, gui.InputTypeCamera, "0", options)
	liveWindow1.Run()

	// 示例2：明确指定为视频文件
	fmt.Println("\n🎬 示例2：明确指定为视频文件")
	fmt.Println("   inputType: gui.InputTypeFile")
	fmt.Println("   inputPath: 'test.mp4'")
	liveWindow2 := gui.NewYOLOLiveWindowWithType(detector, gui.InputTypeFile, "test.mp4", options)
	liveWindow2.Run()

	// 示例3：明确指定为RTSP流
	fmt.Println("\n🎬 示例3：明确指定为RTSP流")
	fmt.Println("   inputType: gui.InputTypeRTSP")
	fmt.Println("   inputPath: 'rtsp://192.168.1.100:554/stream'")
	liveWindow3 := gui.NewYOLOLiveWindowWithType(detector, gui.InputTypeRTSP, "rtsp://192.168.1.100:554/stream", options)
	liveWindow3.Run()

	// 示例4：明确指定为屏幕录制
	fmt.Println("\n🎬 示例4：明确指定为屏幕录制")
	fmt.Println("   inputType: gui.InputTypeScreen")
	fmt.Println("   inputPath: 'desktop'")
	liveWindow4 := gui.NewYOLOLiveWindowWithType(detector, gui.InputTypeScreen, "desktop", options)
	liveWindow4.Run()

	// 示例5：明确指定为RTMP流
	fmt.Println("\n🎬 示例5：明确指定为RTMP流")
	fmt.Println("   inputType: gui.InputTypeRTMP")
	fmt.Println("   inputPath: 'rtmp://server.com/live/stream'")
	liveWindow5 := gui.NewYOLOLiveWindowWithType(detector, gui.InputTypeRTMP, "rtmp://server.com/live/stream", options)
	liveWindow5.Run()

	fmt.Println("\n✅ 明确输入源类型示例完成！")
	fmt.Println("💡 优势:")
	fmt.Println("   - 避免字符串解析歧义")
	fmt.Println("   - 代码更清晰易读")
	fmt.Println("   - 支持更多输入源类型")
	fmt.Println("   - 更好的错误处理")
}
