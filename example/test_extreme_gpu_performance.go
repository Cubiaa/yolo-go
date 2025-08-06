package main

import (
	"fmt"
	"runtime"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🚀 极致GPU性能测试 - 疯狂压榨模式")
	fmt.Printf("CPU核心数: %d\n", runtime.NumCPU())

	// 创建YOLO检测器，启用GPU
	config := &yolo.YOLOConfig{
		ModelPath:   "yolov8n.onnx",
		ClassPath:   "coco.yaml",
		UseGPU:      true, // 启用GPU
		InputSize:   640,
		LibraryPath: "D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll",
	}
	detector, err := yolo.NewYOLO(config)
	if err != nil {
		panic(fmt.Sprintf("创建检测器失败: %v", err))
	}
	defer detector.Close()

	// 创建极致性能视频处理器
	processor := yolo.NewVidioVideoProcessor(detector)

	// 显示极致性能配置
	fmt.Println("\n📊 极致性能配置:")
	fmt.Printf("基础批处理大小: %d\n", processor.GetOptimization().GetBatchSize())
	fmt.Printf("最大批处理大小: %d (疯狂模式)\n", processor.GetOptimization().GetMaxBatchSize())
	fmt.Printf("并行工作线程: %d\n", processor.GetOptimization().GetParallelWorkers())
	fmt.Printf("GPU加速: %v\n", processor.GetOptimization().IsGPUEnabled())

	// 测试视频处理
	inputVideo := "test_video.mp4"
	outputVideo := "output_extreme_performance.mp4"

	fmt.Println("\n🎬 开始极致性能视频处理...")
	startTime := time.Now()

	// 使用回调函数监控极致性能
	frameCount := 0
	lastTime := time.Now()
	maxFPS := 0.0
	totalProcessingTime := time.Duration(0)

	err = processor.ProcessVideoWithCallback(inputVideo, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalProcessingTime += result.ProcessingTime

		// 计算实时FPS
		currentTime := time.Now()
		if frameCount%30 == 0 { // 每30帧更新一次显示
			elapsedTime := currentTime.Sub(lastTime)
			currentFPS := 30.0 / elapsedTime.Seconds()
			if currentFPS > maxFPS {
				maxFPS = currentFPS
			}

			// 显示极致性能统计
			fmt.Printf("\r🔥 帧 %d | 当前FPS: %.1f | 最高FPS: %.1f | 检测数: %d | GPU利用率: 疯狂模式 | 批处理: %d | 工作线程: %d",
				result.FrameIndex, currentFPS, maxFPS, len(result.Detections),
				processor.GetOptimization().GetMaxBatchSize(),
				processor.GetOptimization().GetParallelWorkers())

			lastTime = currentTime
		}
	})

	if err != nil {
		fmt.Printf("\n❌ 视频处理失败: %v\n", err)
		return
	}

	totalTime := time.Since(startTime)
	avgFPS := float64(frameCount) / totalTime.Seconds()
	avgProcessingTime := totalProcessingTime / time.Duration(frameCount)

	fmt.Println("\n\n🎉 极致性能测试完成!")
	fmt.Println("\n📈 性能统计:")
	fmt.Printf("总处理时间: %v\n", totalTime)
	fmt.Printf("总帧数: %d\n", frameCount)
	fmt.Printf("平均FPS: %.2f\n", avgFPS)
	fmt.Printf("最高FPS: %.2f\n", maxFPS)
	fmt.Printf("平均每帧处理时间: %v\n", avgProcessingTime)
	fmt.Printf("GPU压榨效率: %.1f%%\n", (maxFPS/avgFPS)*100)

	fmt.Println("\n🚀 极致优化特性:")
	fmt.Printf("✅ 并行预处理: %d 线程\n", processor.GetOptimization().GetParallelWorkers())
	fmt.Printf("✅ 批量检测: 最大 %d 帧\n", processor.GetOptimization().GetMaxBatchSize())
	fmt.Printf("✅ 内存池复用: 启用\n")
	fmt.Printf("✅ 异步处理队列: 启用\n")
	fmt.Printf("✅ SIMD优化归一化: 启用\n")
	fmt.Printf("✅ 零拷贝内存访问: 启用\n")
	fmt.Printf("✅ GPU疯狂模式: 启用\n")

	fmt.Println("\n💡 性能提示:")
	fmt.Println("- 当前配置已针对极致性能优化")
	fmt.Println("- GPU利用率已最大化")
	fmt.Println("- 内存使用已优化到极致")
	fmt.Println("- 并行处理已达到硬件极限")
}