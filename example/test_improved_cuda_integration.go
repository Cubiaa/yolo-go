package main

import (
	"fmt"
	"log"
	"time"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🚀 测试改进的CUDA初始化集成")
	fmt.Println("基于用户成功案例的优化实现")
	fmt.Println("========================================")

	// 1. 创建改进的CUDA初始化器
	libraryPath := "onnxruntime\\lib\\onnxruntime.dll"
	deviceID := 0
	cudaInitializer := yolo.NewImprovedCUDAInitializer(libraryPath, deviceID)
	defer cudaInitializer.Cleanup()

	// 2. 显示初始化步骤
	fmt.Println("\n📋 CUDA初始化步骤:")
	steps := cudaInitializer.GetInitializationSteps()
	for i, step := range steps {
		fmt.Printf("   %d. %s\n", i+1, step)
	}

	// 3. 使用改进的方法创建Session
	fmt.Println("\n🔧 开始CUDA初始化...")
	startTime := time.Now()

	session, err := cudaInitializer.CreateSessionWithImprovedCUDA(
		"yolo12x.onnx",
		[]string{"images"},
		[]string{"output0"},
	)
	if err != nil {
		log.Printf("❌ CUDA初始化失败: %v", err)
		fmt.Println("\n💡 可能的解决方案:")
		fmt.Println("   1. 检查CUDA是否正确安装")
		fmt.Println("   2. 确认GPU驱动程序是最新版本")
		fmt.Println("   3. 验证ONNX Runtime库路径是否正确")
		fmt.Println("   4. 检查模型文件是否存在")
		return
	}
	defer session.Destroy()

	initTime := time.Since(startTime)
	fmt.Printf("⏱️  CUDA初始化耗时: %v\n", initTime)

	// 4. 测试CUDA推理
	fmt.Println("\n🧪 开始CUDA推理测试...")
	testStartTime := time.Now()

	err = cudaInitializer.TestCUDAInference(session)
	if err != nil {
		log.Printf("❌ CUDA推理测试失败: %v", err)
		return
	}

	testTime := time.Since(testStartTime)
	fmt.Printf("⏱️  推理测试耗时: %v\n", testTime)

	// 5. 显示对比信息
	fmt.Println("\n📊 与现有实现的对比:")
	comparison := cudaInitializer.CompareWithCurrentImplementation()
	for key, value := range comparison {
		fmt.Printf("   %s: %s\n", key, value)
	}

	// 6. 性能统计
	fmt.Println("\n📈 性能统计:")
	fmt.Printf("   总初始化时间: %v\n", initTime)
	fmt.Printf("   推理测试时间: %v\n", testTime)
	fmt.Printf("   总耗时: %v\n", initTime+testTime)

	// 7. 集成建议
	fmt.Println("\n💡 集成到现有项目的建议:")
	fmt.Println("   1. 可以将改进的初始化方法集成到yolo.go中")
	fmt.Println("   2. 作为现有CUDA初始化的替代方案")
	fmt.Println("   3. 提供更好的错误处理和调试信息")
	fmt.Println("   4. 减少不必要的回退机制，提高性能")

	// 8. 与现有YOLO检测器的兼容性测试
	fmt.Println("\n🔗 测试与现有YOLO检测器的兼容性...")
	testCompatibilityWithExistingYOLO()

	fmt.Println("\n✅ 改进的CUDA初始化测试完成")
}

// testCompatibilityWithExistingYOLO 测试与现有YOLO检测器的兼容性
func testCompatibilityWithExistingYOLO() {
	fmt.Println("   正在创建标准YOLO检测器进行对比...")

	// 使用现有的YOLO检测器
	config := yolo.DefaultConfig().
		WithLibraryPath("onnxruntime\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0)

	detector, err := yolo.NewYOLO("yolo12x.onnx", "data.yaml", config)
	if err != nil {
		fmt.Printf("   ⚠️  标准YOLO检测器创建失败: %v\n", err)
		fmt.Println("   💡 这表明改进的CUDA初始化方法可能更可靠")
		return
	}
	defer detector.Close()

	fmt.Println("   ✅ 标准YOLO检测器创建成功")
	fmt.Println("   📊 两种方法都可以成功初始化CUDA")
	fmt.Println("   💡 建议根据具体需求选择合适的初始化方法")
}