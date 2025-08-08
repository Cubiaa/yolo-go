package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🔍 GPU 使用情况验证程序")
	fmt.Println("==========================")

	// 显示系统信息
	fmt.Printf("💻 CPU核心数: %d\n", runtime.NumCPU())
	fmt.Printf("🎯 模型: yolo12x.onnx\n")
	fmt.Printf("📹 视频: test.mp4\n")
	fmt.Println()

	// 创建检测器 - 使用更合理的批处理大小
	config := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0)

	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", config)
	if err != nil {
		log.Fatal("创建检测器失败:", err)
	}
	defer detector.Close()

	// 验证 GPU 状态
	processor := detector.GetVideoProcessor()
	optimization := processor.GetOptimization()

	fmt.Println("📊 GPU 状态验证:")
	fmt.Printf("   GPU启用: %v\n", optimization.IsGPUEnabled())
	fmt.Printf("   CUDA启用: %v\n", optimization.IsCUDAEnabled())
	fmt.Printf("   CUDA设备ID: %d\n", optimization.GetCUDADeviceID())
	fmt.Printf("   批处理大小: %d\n", optimization.GetBatchSize())
	fmt.Printf("   并行工作线程: %d\n", optimization.GetParallelWorkers())
	fmt.Println()

	// 显示 CUDA 性能指标
	if optimization.IsCUDAEnabled() {
		metrics := optimization.GetCUDAPerformanceMetrics()
		fmt.Println("🚀 CUDA 性能指标:")
		for key, value := range metrics {
			fmt.Printf("   %s: %v\n", key, value)
		}
		fmt.Println()
	}

	// 创建检测选项
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithDrawLabels(true).
		WithConfThreshold(0.5).
		WithIOUThreshold(0.4).
		WithShowFPS(true)

	fmt.Println("🎬 开始处理视频 (请观察任务管理器 GPU 计算引擎)...")
	fmt.Println("💡 提示: 在任务管理器中切换到 GPU 的 '计算 (Compute_0)' 标签页")
	fmt.Println()

	startTime := time.Now()
	frameCount := 0
	var totalProcessingTime time.Duration

	// 处理视频并监控性能
	results, err := detector.Detect("test.mp4", options, func(result yolo.VideoDetectionResult) {
		frameCount++
		totalProcessingTime += result.Timestamp

		// 每 30 帧显示一次进度和性能统计
		if frameCount%30 == 0 {
			elapsed := time.Since(startTime)
			currentFPS := float64(frameCount) / elapsed.Seconds()
			avgFrameTime := totalProcessingTime / time.Duration(frameCount)

			fmt.Printf("📊 帧 %d | FPS: %.1f | 平均帧时间: %v | 检测对象: %d\n",
				frameCount, currentFPS, avgFrameTime, len(result.Detections))

			// 显示 GPU 状态
			if optimization.IsCUDAEnabled() {
				metrics := optimization.GetCUDAPerformanceMetrics()
				if enabled, ok := metrics["enabled"].(bool); ok && enabled {
					fmt.Printf("   🚀 GPU 正在工作 | 延迟: %v\n", metrics["latency_ms"])
				}
			}
		}
	})

	if err != nil {
		log.Printf("❌ 视频处理失败: %v", err)
		return
	}

	totalTime := time.Since(startTime)
	avgFPS := float64(frameCount) / totalTime.Seconds()

	fmt.Println("\n🎉 处理完成!")
	fmt.Printf("📈 性能统计:\n")
	fmt.Printf("   总帧数: %d\n", frameCount)
	fmt.Printf("   总时间: %v\n", totalTime)
	fmt.Printf("   平均FPS: %.2f\n", avgFPS)
	fmt.Printf("   总检测对象: %d\n", len(results.Detections))

	fmt.Println("\n💡 GPU 使用情况分析:")
	fmt.Println("1. 如果看到 '计算 (Compute_0)' 有活动，说明 GPU 在工作")
	fmt.Println("2. 如果 GPU 使用率低，可能是因为:")
	fmt.Println("   - 视频解码在 CPU 成为瓶颈")
	fmt.Println("   - 输入分辨率较低 (720p)")
	fmt.Println("   - 批处理大小需要调整")
	fmt.Println("3. 要增加 GPU 使用率，可以:")
	fmt.Println("   - 使用更高分辨率视频")
	fmt.Println("   - 增加输入尺寸 (1024x1024)")
	fmt.Println("   - 使用批量检测模式")

	// 保存结果
	fmt.Println("\n💾 保存结果...")
	err = results.Save("gpu_verification_output.mp4")
	if err != nil {
		log.Printf("❌ 保存失败: %v", err)
		return
	}
	fmt.Println("✅ 结果已保存为: gpu_verification_output.mp4")
}
