package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🚀 启动高性能YOLO检测器...")

	// 🔧 使用高端GPU优化配置，自动检测显存并调整参数
	config := yolo.HighEndGPUOptimizedConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll")

	// 📊 显示系统信息
	fmt.Printf("💻 CPU核心数: %d\n", runtime.NumCPU())
	fmt.Printf("🚀 GPU加速: %v\n", config.UseGPU)
	fmt.Printf("⚡ CUDA加速: %v\n", config.UseCUDA)

	// 创建检测器
	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", config)
	if err != nil {
		log.Fatal("创建YOLO检测器失败:", err)
	}
	defer detector.Close()

	// 🎯 创建优化的检测选项
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithLineWidth(3).
		WithFontSize(20).
		WithBoxColor("green").
		WithConfThreshold(0.5). // 置信度阈值
		WithIOUThreshold(0.4)   // NMS阈值

	// 📈 显示优化状态
	if detector.GetVideoProcessor() != nil {
		processor := detector.GetVideoProcessor()
		if optimization := processor.GetOptimization(); optimization != nil {
			fmt.Printf("🚀 GPU优化状态:\n")
			fmt.Printf("   - GPU启用: %v\n", optimization.IsGPUEnabled())
			fmt.Printf("   - CUDA启用: %v\n", optimization.IsCUDAEnabled())
			fmt.Printf("   - 批处理大小: %d\n", optimization.GetBatchSize())
			fmt.Printf("   - 并行工作线程: %d\n", optimization.GetParallelWorkers())
		}
	}

	// ⏱️ 开始计时
	startTime := time.Now()
	fmt.Println("\n🎬 开始处理视频...")

	// 🎯 执行检测
	result, err := detector.Detect("test.mp4", options)
	if err != nil {
		fmt.Printf("❌ 检测失败: %v\n", err)
		return
	}

	// 📊 显示检测结果统计
	processingTime := time.Since(startTime)
	fmt.Printf("\n✅ 检测完成!\n")
	fmt.Printf("📊 检测统计:\n")
	fmt.Printf("   - 总检测对象: %d\n", len(result.Detections))
	fmt.Printf("   - 处理时间: %v\n", processingTime)

	// 如果有视频结果，显示更详细的统计
	if len(result.VideoResults) > 0 {
		totalFrames := len(result.VideoResults)
		fps := float64(totalFrames) / processingTime.Seconds()
		fmt.Printf("   - 总帧数: %d\n", totalFrames)
		fmt.Printf("   - 平均FPS: %.2f\n", fps)
		fmt.Printf("   - 每帧平均时间: %.2fms\n", 1000.0/fps)
	}

	// 💾 保存结果
	fmt.Println("\n💾 保存处理结果...")
	saveStartTime := time.Now()
	err = result.Save("123.mp4")
	if err != nil {
		fmt.Printf("❌ 保存失败: %v\n", err)
		return
	}

	saveTime := time.Since(saveStartTime)
	totalTime := time.Since(startTime)

	fmt.Printf("✅ 保存完成!\n")
	fmt.Printf("📊 完整统计:\n")
	fmt.Printf("   - 检测时间: %v\n", processingTime)
	fmt.Printf("   - 保存时间: %v\n", saveTime)
	fmt.Printf("   - 总时间: %v\n", totalTime)

	// 🎯 显示性能建议
	yolo.HighEndGPUPerformanceTips()

	// 📈 显示GPU基准测试配置
	benchmarkConfig := yolo.GetOptimalGPUSettings()
	fmt.Printf("\n📈 当前GPU配置: %s\n", benchmarkConfig["gpu_tier"])
	fmt.Printf("🎯 预期性能: %s\n", benchmarkConfig["expected_fps"])

	fmt.Println("\n🎉 处理完成！输出文件: 123.mp4")
}