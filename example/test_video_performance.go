package main

// 🔧 重要说明：模型输入尺寸匹配问题修复
// 
// 问题：yolo12x.onnx模型固定期望640x640输入，但代码尝试使用416x416等其他尺寸
// 错误：Got invalid dimensions for input: images for the following indices
//       index: 2 Got: 416 Expected: 640
//       index: 3 Got: 416 Expected: 640
//
// 解决方案：
// 1. 对于yolo12x.onnx，只使用640x640输入尺寸
// 2. 如需测试其他尺寸，请使用对应的模型文件：
//    - 416x416: yolo11n-416.onnx 或 yolo8n-416.onnx
//    - 832x832: yolo11l-832.onnx 或 yolo8l-832.onnx  
//    - 1280x1280: yolo11x-1280.onnx 或 yolo8x-1280.onnx

import (
	"fmt"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO-Go 视频性能测试 (修复版) ===")
	fmt.Println("🎬 测试视频处理性能和GPU利用率...")
	fmt.Println("🔧 已修复：模型输入尺寸匹配问题\n")

	// 检查GPU支持
	fmt.Println("🔍 检查GPU支持状态:")
	yolo.CheckGPUSupport()
	fmt.Println()

	// 测试不同输入尺寸的性能
	testVideoPerformance()

	fmt.Println("\n✅ 视频性能测试完成！")
}

func testVideoPerformance() {
	// 注意：yolo12x.onnx模型固定使用640x640输入尺寸
	// 如果需要测试其他尺寸，请使用对应的模型文件
	inputSizes := []int{640} // 仅测试640，匹配yolo12x.onnx
	videoPath := "test.mp4" // 请确保有这个测试视频文件

	fmt.Println("📊 测试yolo12x.onnx模型性能 (640x640):")
	fmt.Println("💡 请在另一个终端运行 'nvidia-smi -l 1' 监控GPU使用率")
	fmt.Println("⚠️  注意：yolo12x.onnx模型仅支持640x640输入尺寸")
	fmt.Println("💡 如需测试其他尺寸，请使用对应的模型文件 (如yolo11n-416.onnx)\n")

	for _, size := range inputSizes {
		fmt.Printf("🧪 测试输入尺寸: %dx%d\n", size, size)
		testSingleVideoConfig(videoPath, size)
		fmt.Println("---")
	}

	// 提供其他尺寸测试的建议
	fmt.Println("\n💡 测试其他输入尺寸的建议:")
	fmt.Println("   416x416: 使用 yolo11n-416.onnx 或 yolo8n-416.onnx")
	fmt.Println("   832x832: 使用 yolo11l-832.onnx 或 yolo8l-832.onnx")
	fmt.Println("   1280x1280: 使用 yolo11x-1280.onnx 或 yolo8x-1280.onnx")
}

func testSingleVideoConfig(videoPath string, inputSize int) {
	// 创建配置 - 注意：yolo12x.onnx模型固定使用640x640输入
	// 如果inputSize不是640，我们需要跳过或使用640
	if inputSize != 640 {
		fmt.Printf("⚠️  跳过测试：yolo12x.onnx模型仅支持640x640输入，当前请求: %dx%d\n", inputSize, inputSize)
		return
	}
	
	config := yolo.DefaultConfig().
		WithGPU(true).
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithInputSize(640) // 固定使用640，匹配yolo12x.onnx模型

	// 创建检测器
	detector, err := yolo.NewYOLO("yolo12x.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("❌ 创建检测器失败: %v\n", err)
		return
	}
	defer detector.Close()

	// 创建检测选项
	options := yolo.DefaultDetectionOptions().
		WithConfThreshold(0.25).
		WithIOUThreshold(0.45).
		WithDrawBoxes(true).
		WithDrawLabels(true)

	// 开始计时
	startTime := time.Now()
	fmt.Printf("⏱️  开始处理视频: %s\n", time.Now().Format("15:04:05"))
	fmt.Printf("📈 请观察nvidia-smi中的GPU利用率\n")

	// 设置检测器的运行时配置
	detector.SetRuntimeConfig(options)

	// 处理视频
	results, err := detector.DetectVideo(videoPath, false)
	if err != nil {
		fmt.Printf("❌ 视频处理失败: %v\n", err)
		return
	}

	// 计算处理时间
	processingTime := time.Since(startTime)
	fmt.Printf("✅ 处理完成: %s\n", time.Now().Format("15:04:05"))
	fmt.Printf("⏱️  总处理时间: %v\n", processingTime)

	// 统计总检测对象数
	totalDetections := 0
	for _, result := range results {
		totalDetections += len(result.Detections)
	}
	fmt.Printf("🎯 检测到对象: %d\n", totalDetections)

	// 性能分析
	analyzePerformance(inputSize, processingTime)
}

func analyzePerformance(inputSize int, processingTime time.Duration) {
	fmt.Printf("📊 性能分析:\n")
	fmt.Printf("   输入尺寸: %dx%d (yolo12x.onnx固定尺寸)\n", inputSize, inputSize)
	fmt.Printf("   处理时间: %v\n", processingTime)

	// 高性能GPU + yolo12x.onnx (640x640) 预期性能
	expectedTime := 3 * time.Second
	expectedGPUUsage := "30-50%"

	fmt.Printf("   预期时间: %v (高性能GPU + yolo12x.onnx)\n", expectedTime)
	fmt.Printf("   预期GPU使用率: %s\n", expectedGPUUsage)

	// 性能评估
	if processingTime <= expectedTime {
		fmt.Printf("   ✅ 性能正常\n")
	} else if processingTime <= expectedTime*2 {
		fmt.Printf("   ⚠️  性能略慢\n")
	} else {
		fmt.Printf("   ❌ 性能异常慢 (%.1fx slower)\n", float64(processingTime)/float64(expectedTime))
		fmt.Printf("   💡 可能原因:\n")
		fmt.Printf("      - GPU未正确启用\n")
		fmt.Printf("      - 系统资源不足\n")
		fmt.Printf("      - 视频文件在网络存储\n")
		fmt.Printf("      - CUDA/cuDNN版本问题\n")
	}
}

// 添加实时监控函数
func printMonitoringInstructions() {
	fmt.Println("\n📈 实时监控指南:")
	fmt.Println("\n1. 打开新的PowerShell窗口")
	fmt.Println("2. 运行命令: nvidia-smi -l 1")
	fmt.Println("3. 观察GPU利用率变化")
	fmt.Println("\n🎯 正常GPU利用率范围:")
	fmt.Println("   416x416: 15-30%")
	fmt.Println("   640x640: 30-50%")
	fmt.Println("   832x832: 50-70%")
	fmt.Println("   1280x1280: 70-90%")
	fmt.Println("\n⚠️  如果GPU利用率持续低于15%，说明存在问题")
}