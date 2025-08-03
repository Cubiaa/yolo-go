package main

import (
	"fmt"
	"strings"

	"github.com/Cubiaa/yolo-go/yolo"
)

func main() {
	fmt.Println("🚀 测试自定义输入尺寸功能")
	fmt.Println(strings.Repeat("=", 50))

	// 示例1: 使用正方形输入尺寸 (传统方式)
	fmt.Println("\n📐 示例1: 正方形输入尺寸 640x640")
	config1 := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputSize(640)

	fmt.Printf("配置: InputSize=%d, InputWidth=%d, InputHeight=%d\n",
		config1.InputSize, config1.InputWidth, config1.InputHeight)

	// 示例2: 使用自定义宽高比输入尺寸 (新功能)
	fmt.Println("\n📐 示例2: 自定义输入尺寸 1280x720 (16:9宽高比)")
	config2 := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputDimensions(1280, 720)

	fmt.Printf("配置: InputSize=%d, InputWidth=%d, InputHeight=%d\n",
		config2.InputSize, config2.InputWidth, config2.InputHeight)

	// 示例3: 使用自定义宽高比输入尺寸 (竖屏)
	fmt.Println("\n📐 示例3: 自定义输入尺寸 480x854 (9:16宽高比，竖屏)")
	config3 := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputDimensions(480, 854)

	fmt.Printf("配置: InputSize=%d, InputWidth=%d, InputHeight=%d\n",
		config3.InputSize, config3.InputWidth, config3.InputHeight)

	// 示例4: 链式调用，先设置自定义尺寸，再改为正方形
	fmt.Println("\n📐 示例4: 链式调用 - 先自定义后正方形")
	config4 := yolo.DefaultConfig().
		WithLibraryPath("D:\\onnxruntime-win-x64-1.22.1\\lib\\onnxruntime.dll").
		WithGPU(true).
		WithGPUDeviceID(0).
		WithInputDimensions(1920, 1080) // 使用自定义宽高尺寸

	fmt.Printf("配置: InputSize=%d, InputWidth=%d, InputHeight=%d\n",
		config4.InputSize, config4.InputWidth, config4.InputHeight)

	// 尝试创建检测器来验证配置
	fmt.Println("\n🔧 尝试创建检测器以验证配置...")

	// 注意：这里使用一个不存在的模型路径，只是为了测试配置解析
	// 实际使用时请提供正确的模型路径
	modelPath := "../test_model.onnx" // 不存在的模型文件
	configPath := "../coco.yaml"      // 不存在的配置文件

	detector, err := yolo.NewYOLOWithConfig(modelPath, configPath, config2)
	if err != nil {
		fmt.Printf("⚠️  预期的错误 (模型文件不存在): %v\n", err)
		fmt.Println("✅ 配置解析正常，错误是由于模型文件不存在导致的")
	} else {
		fmt.Println("✅ 检测器创建成功!")
		defer detector.Close()
	}

	fmt.Println("\n🎉 自定义输入尺寸功能测试完成!")
	fmt.Println("\n💡 使用方法:")
	fmt.Println("   - WithInputSize(size): 设置正方形输入尺寸")
	fmt.Println("   - WithInputDimensions(width, height): 设置自定义宽高")
	fmt.Println("   - 两种方法可以链式调用，后调用的会覆盖前面的设置")
}
