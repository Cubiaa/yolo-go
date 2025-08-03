package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/Cubiaa/yolo-go/gui"
	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🔍 YOLO模型自动检测测试")
	fmt.Printf("💻 系统信息: %d CPU核心, %s架构\n", runtime.NumCPU(), runtime.GOARCH)

	// 1. 使用默认配置（现已集成智能模型适配）
	fmt.Println("\n=== 默认配置（自动检测模型） ===")
	defaultConfig := yolo.DefaultConfig()
	fmt.Printf("默认配置: 输入尺寸=%d, GPU启用=%t\n", defaultConfig.InputSize, defaultConfig.UseGPU)

	// 2. 创建YOLO检测器（使用默认配置）
	fmt.Println("\n=== 创建检测器 ===")
	detector, err := yolo.NewYOLO("../yolo12x.onnx", "../coco.yaml", 
		yolo.DefaultConfig().WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").WithGPU(true).WithGPUDeviceID(0))
	if err != nil {
		log.Printf("创建检测器失败: %v", err)
		return
	}
	defer detector.Close()

	// 3. 测试不同模型的自动检测
	fmt.Println("\n=== 测试不同模型的输入尺寸检测 ===")
	testModels := []string{
		"yolo11n-640.onnx",
		"yolo8s-416.onnx", 
		"yolo12x-1280.onnx",
		"custom-model-512.onnx",
		"unknown-model.onnx",
	}

	for _, modelName := range testModels {
		config := yolo.AutoDetectInputSizeConfig(modelName)
		fmt.Printf("📁 %s -> 输入尺寸: %d\n", modelName, config.InputSize)
	}

	// 4. 检测选项配置
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithDrawLabels(true).
		WithConfThreshold(0.25). // 降低置信度阈值
		WithIOUThreshold(0.45).  // 调整IOU阈值
		WithShowFPS(true)        // 显示FPS

	fmt.Printf("🎯 检测配置: 置信度阈值=%.2f, IOU阈值=%.2f\n", 
		options.ConfThreshold, options.IOUThreshold)

	// 5. 启动实时检测窗口
	fmt.Println("\n🎬 启动自动适配模型的视频检测窗口...")
	window := gui.NewYOLOLiveWindow(detector, gui.InputTypeFile, "../test.mp4", options)
	window.Run()

	fmt.Println("✅ 自动检测模型测试完成！")
	fmt.Println("💡 优势:")
	fmt.Println("   - 自动检测模型输入尺寸，无需手动配置")
	fmt.Println("   - 支持多种YOLO模型版本")
	fmt.Println("   - 智能回退到默认配置")
	fmt.Println("   - 实时显示模型信息")
}