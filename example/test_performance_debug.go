package main

import (
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("=== YOLO-Go 性能诊断工具 ===")
	fmt.Println("🔍 正在检查系统配置和GPU状态...\n")

	// 1. 系统信息检查
	fmt.Println("📊 系统信息:")
	fmt.Printf("   CPU核心数: %d\n", runtime.NumCPU())
	fmt.Printf("   Go版本: %s\n", runtime.Version())
	fmt.Printf("   操作系统: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Println()

	// 2. GPU支持检查
	fmt.Println("🚀 GPU支持检查:")
	yolo.CheckGPUSupport()
	fmt.Println()

	// 3. 测试不同配置的性能
	testConfigurations()

	fmt.Println("\n✅ 性能诊断完成！")
	fmt.Println("💡 如果GPU利用率仍然很低，请检查:")
	fmt.Println("   1. 视频分辨率是否过低")
	fmt.Println("   2. 输入尺寸是否设置正确")
	fmt.Println("   3. 是否有其他程序占用GPU")
	fmt.Println("   4. CUDA/cuDNN版本是否匹配")
}

func testConfigurations() {
	fmt.Println("⚡ 测试不同配置的性能:")

	// 测试配置列表
	configs := []struct {
		name   string
		config *yolo.YOLOConfig
	}{
		{"默认配置", yolo.DefaultConfig()},
		{"GPU极限配置", yolo.ExtremePerformanceConfig()},
		{"GPU最大性能", yolo.MaxPerformanceGPUConfig()},
		{"CPU最大性能", yolo.MaxPerformanceCPUConfig()},
	}

	for _, cfg := range configs {
		fmt.Printf("\n🧪 测试配置: %s\n", cfg.name)
		testSingleConfig(cfg.config)
	}
}

func testSingleConfig(config *yolo.YOLOConfig) {
	// 显示配置详情
	fmt.Printf("   配置详情:\n")
	fmt.Printf("     输入尺寸: %dx%d\n", config.InputSize, config.InputSize)
	fmt.Printf("     使用GPU: %v\n", config.UseGPU)
	fmt.Printf("     GPU设备ID: %d\n", config.GPUDeviceID)
	fmt.Printf("     库路径: %s\n", config.LibraryPath)

	// 尝试创建检测器
	start := time.Now()
	detector, err := yolo.NewYOLO("yolo12x.onnx", "coco.yaml", config)
	if err != nil {
		fmt.Printf("   ❌ 创建检测器失败: %v\n", err)
		return
	}
	defer detector.Close()

	creationTime := time.Since(start)
	fmt.Printf("   ✅ 检测器创建成功，耗时: %v\n", creationTime)

	// 如果有测试视频，可以进行实际性能测试
	// 这里只是演示配置是否能正常工作
	fmt.Printf("   💡 配置验证完成\n")
}

// 添加GPU监控函数
func monitorGPUUsage() {
	fmt.Println("\n📈 GPU使用率监控:")
	fmt.Println("💡 请在另一个终端运行以下命令来监控GPU使用率:")
	fmt.Println("   nvidia-smi -l 1")
	fmt.Println("\n🎯 预期GPU利用率:")
	fmt.Println("   YOLOv12x + 640x640输入: 30-60%")
	fmt.Println("   YOLOv12x + 1280x1280输入: 60-90%")
	fmt.Println("   如果利用率低于30%，可能存在性能瓶颈")
}

// 添加性能优化建议
func printOptimizationTips() {
	fmt.Println("\n🚀 性能优化建议:")
	fmt.Println("\n1. 提高输入尺寸:")
	fmt.Println("   config.WithInputSize(1280)  // 从640提升到1280")
	fmt.Println("\n2. 使用极限配置:")
	fmt.Println("   yolo.ExtremePerformanceConfig()")
	fmt.Println("\n3. 批量处理:")
	fmt.Println("   同时处理多个视频或图像")
	fmt.Println("\n4. 检查系统瓶颈:")
	fmt.Println("   - CPU使用率是否过高")
	fmt.Println("   - 内存是否充足")
	fmt.Println("   - 存储设备是否为SSD")
	fmt.Println("   - 视频文件是否在本地")
}