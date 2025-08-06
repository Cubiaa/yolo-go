package main

import (
	"fmt"
	"log"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🚀 测试自动CUDA加速 - WithGPU(true)现在自动启用CUDA")

	// 现在WithGPU(true)会自动启用CUDA加速
	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", 
		yolo.DefaultConfig().
			WithLibraryPath("onnxruntime\\lib\\onnxruntime.dll").
			WithGPU(true)) // 这行现在会自动启用CUDA!

	if err != nil {
		log.Printf("❌ 初始化YOLO检测器失败: %v", err)
		return
	}
	defer detector.Close()

	fmt.Println("✅ YOLO检测器初始化成功")

	// 验证CUDA是否已自动启用
	optimization := detector.GetVideoProcessor().GetOptimization()
	fmt.Printf("📊 配置状态:\n")
	fmt.Printf("   GPU加速: %v\n", optimization.IsGPUEnabled())
	fmt.Printf("   CUDA加速: %v\n", optimization.IsCUDAEnabled())
	fmt.Printf("   CUDA设备ID: %d\n", optimization.GetCUDADeviceID())
	fmt.Printf("   批处理大小: %d\n", optimization.GetBatchSize())
	fmt.Printf("   并行工作线程: %d\n", optimization.GetParallelWorkers())

	// 显示CUDA性能指标
	if optimization.IsCUDAEnabled() {
		metrics := optimization.GetCUDAPerformanceMetrics()
		fmt.Printf("🚀 CUDA性能指标: %+v\n", metrics)
	}

	// 处理视频
	options := yolo.DefaultDetectionOptions().
		WithDrawBoxes(true).
		WithBoxColor("green").
		WithDrawLabels(true).
		WithLabelColor("red").
		WithLineWidth(2).
		WithFontSize(3)

	fmt.Println("🎬 开始处理视频文件 pot.mp4...")
	startTime := time.Now()

	results, err := detector.Detect("pot.mp4", options, func(result yolo.VideoDetectionResult) {
		// 每100帧显示一次进度
		if result.FrameNumber%100 == 0 {
			fmt.Printf("📊 正在处理第 %d 帧...\n", result.FrameNumber)
		}
	})

	if err != nil {
		log.Printf("❌ 视频检测失败: %v", err)
		return
	}

	processingTime := time.Since(startTime)
	fmt.Printf("✅ 视频处理完成，耗时: %v\n", processingTime)

	fmt.Println("💾 正在保存结果视频...")
	err = results.SaveWithAudio("pot_result_auto_cuda.mp4")
	if err != nil {
		log.Printf("❌ 保存视频失败: %v", err)
		return
	}

	fmt.Println("🎉 程序执行完成！结果已保存为 pot_result_auto_cuda.mp4")

	fmt.Println("\n📋 重要说明:")
	fmt.Println("现在使用 WithGPU(true) 会自动启用以下优化:")
	fmt.Println("✅ GPU推理加速")
	fmt.Println("✅ CUDA图像预处理加速")
	fmt.Println("✅ CUDA批处理优化")
	fmt.Println("✅ CUDA内存池优化")
	fmt.Println("✅ 自动回退机制（CUDA不可用时回退到CPU）")
	fmt.Println("\n这意味着您无需手动配置CUDA，WithGPU(true)就能获得最佳性能！")
}