package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"../yolo"
)

func main() {
	fmt.Println("🚀 高端GPU优化测试程序")
	fmt.Println("支持高端GPU (8GB+显存)等高性能显卡")
	fmt.Println("========================================")

	// 检查命令行参数
	if len(os.Args) < 3 {
		fmt.Println("使用方法: go run test_high_end_gpu_optimization.go <模型路径> <视频路径>")
		fmt.Println("示例: go run test_high_end_gpu_optimization.go yolo11n.onnx test_video.mp4")
		return
	}

	modelPath := os.Args[1]
	videoPath := os.Args[2]

	// 显示当前系统信息
	fmt.Printf("💻 CPU核心数: %d\n", runtime.NumCPU())
	fmt.Printf("🎯 模型路径: %s\n", modelPath)
	fmt.Printf("📹 视频路径: %s\n", videoPath)
	fmt.Println()

	// 1. 获取最优GPU设置建议
	fmt.Println("📊 获取GPU优化建议...")
	optimalSettings := yolo.GetOptimalGPUSettings()
	fmt.Printf("优化级别: %s\n", optimalSettings["optimization_level"])
	fmt.Println()

	// 2. 显示高端GPU性能优化建议
	yolo.HighEndGPUPerformanceTips()

	// 3. 创建高端GPU优化配置
	fmt.Println("⚙️ 创建高端GPU优化配置...")
	config := yolo.HighEndGPUOptimizedConfig()

	// 4. 初始化YOLO检测器
	fmt.Println("🔧 初始化YOLO检测器...")
	detector, err := yolo.NewYOLO(modelPath, config)
	if err != nil {
		log.Fatalf("初始化YOLO失败: %v", err)
	}
	defer detector.Close()

	// 5. 创建自适应GPU视频优化实例
	fmt.Println("🚀 创建自适应GPU视频优化实例...")
	optimization := yolo.NewAdaptiveGPUVideoOptimization()
	defer optimization.Close()

	// 6. 显示优化配置信息
	fmt.Println("\n📋 当前优化配置:")
	fmt.Printf("批处理大小: %d\n", optimization.GetBatchSize())
	fmt.Printf("最大批处理: %d\n", optimization.GetMaxBatchSize())
	fmt.Printf("并行工作线程: %d\n", optimization.GetParallelWorkers())
	fmt.Printf("GPU启用: %t\n", optimization.IsGPUEnabled())
	fmt.Printf("CUDA启用: %t\n", optimization.IsCUDAEnabled())
	if optimization.IsCUDAEnabled() {
		cudaMetrics := optimization.GetCUDAPerformanceMetrics()
		fmt.Printf("CUDA设备ID: %v\n", cudaMetrics["device_id"])
		fmt.Printf("CUDA流数量: %v\n", cudaMetrics["stream_count"])
	}
	fmt.Println()

	// 7. 创建视频处理器
	fmt.Println("📹 创建视频处理器...")
	processor, err := yolo.NewVidioVideoProcessor(detector)
	if err != nil {
		log.Fatalf("创建视频处理器失败: %v", err)
	}
	defer processor.Close()

	// 8. 设置优化实例
	processor.SetOptimization(optimization)

	// 9. 开始性能测试
	fmt.Println("🏁 开始高端GPU性能测试...")
	startTime := time.Now()

	// 处理视频并统计性能
	frameCount := 0
	processingTimes := []time.Duration{}

	err = processor.ProcessVideoWithCallback(videoPath, func(result *yolo.VideoDetectionResult) {
		frameStart := time.Now()
		
		frameCount++
		if frameCount%100 == 0 {
			elapsed := time.Since(startTime)
			fps := float64(frameCount) / elapsed.Seconds()
			fmt.Printf("📊 已处理 %d 帧, 当前FPS: %.2f, 检测到 %d 个对象\n", 
				frameCount, fps, len(result.Detections))
			
			// 显示稳定性状态
			stabilityStatus := optimization.GetStabilityStatus()
			fmt.Printf("🔧 稳定性状态: 健康=%t, 成功率=%.2f%%, 平均延迟=%v\n",
				stabilityStatus["is_healthy"],
				stabilityStatus["success_rate"],
				stabilityStatus["avg_latency"])
			
			// 显示队列状态
			asyncQueue, processDone, availableWorkers := optimization.GetQueueStatus()
			fmt.Printf("📈 队列状态: 异步队列=%d, 完成队列=%d, 可用工作线程=%d\n",
				asyncQueue, processDone, availableWorkers)
			fmt.Println()
		}
		
		frameTime := time.Since(frameStart)
		processingTimes = append(processingTimes, frameTime)
	})

	if err != nil {
		log.Fatalf("视频处理失败: %v", err)
	}

	// 10. 计算最终性能统计
	totalTime := time.Since(startTime)
	avgFPS := float64(frameCount) / totalTime.Seconds()

	fmt.Println("\n🎉 高端GPU性能测试完成!")
	fmt.Println("========================================")
	fmt.Printf("📊 总处理帧数: %d\n", frameCount)
	fmt.Printf("⏱️  总处理时间: %v\n", totalTime)
	fmt.Printf("🚀 平均FPS: %.2f\n", avgFPS)

	// 计算处理时间统计
	if len(processingTimes) > 0 {
		var totalFrameTime time.Duration
		minTime := processingTimes[0]
		maxTime := processingTimes[0]
		
		for _, t := range processingTimes {
			totalFrameTime += t
			if t < minTime {
				minTime = t
			}
			if t > maxTime {
				maxTime = t
			}
		}
		
		avgFrameTime := totalFrameTime / time.Duration(len(processingTimes))
		fmt.Printf("📈 平均单帧处理时间: %v\n", avgFrameTime)
		fmt.Printf("⚡ 最快单帧处理时间: %v\n", minTime)
		fmt.Printf("🐌 最慢单帧处理时间: %v\n", maxTime)
	}

	// 11. 显示最终稳定性报告
	fmt.Println("\n📋 最终稳定性报告:")
	stabilityStatus := optimization.GetStabilityStatus()
	for key, value := range stabilityStatus {
		fmt.Printf("%s: %v\n", key, value)
	}

	// 12. 显示GC统计
	fmt.Println("\n🗑️ 垃圾回收统计:")
	gcStats := optimization.GetGCStats()
	for key, value := range gcStats {
		fmt.Printf("%s: %v\n", key, value)
	}

	// 13. 性能对比和建议
	fmt.Println("\n💡 性能分析和建议:")
	expectedFPS := optimalSettings["expected_fps"].(string)
	targetTime := optimalSettings["target_time"].(string)
	fmt.Printf("预期性能: %s\n", expectedFPS)
	fmt.Printf("目标时间: %s\n", targetTime)
	
	if avgFPS >= 300 {
		fmt.Println("🎉 性能优秀! 已达到高端GPU预期性能")
	} else if avgFPS >= 200 {
		fmt.Println("✅ 性能良好! 接近高端GPU预期性能")
	} else if avgFPS >= 100 {
		fmt.Println("⚠️ 性能一般，建议检查:")
		fmt.Println("   - 确保使用最新CUDA驱动")
		fmt.Println("   - 关闭不必要的后台程序")
		fmt.Println("   - 检查GPU温度和功耗限制")
	} else {
		fmt.Println("❌ 性能较低，建议:")
		fmt.Println("   - 检查CUDA是否正确安装")
		fmt.Println("   - 确认GPU驱动版本")
		fmt.Println("   - 考虑降低批处理大小")
		fmt.Println("   - 检查系统资源占用")
	}

	fmt.Println("\n🏁 测试完成!")
}