package main

import (
	"fmt"
	"log"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	fmt.Println("🚀 测试成功的CUDA初始化方式")
	fmt.Println("基于用户提供的成功案例")

	// 设置 ONNX Runtime 库路径
	ort.SetSharedLibraryPath(`onnxruntime/lib/onnxruntime.dll`)

	// 初始化环境
	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// 创建 SessionOptions
	opts, err := ort.NewSessionOptions()
	if err != nil {
		panic(err)
	}
	defer opts.Destroy()

	// 配置 CUDA Provider
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		fmt.Println("CUDA Provider 创建失败:", err)
		return
	}
	defer cudaOpts.Destroy()

	err = cudaOpts.Update(map[string]string{
		"device_id": "0",
	})
	if err != nil {
		fmt.Println("CUDA 配置失败:", err)
		return
	}

	err = opts.AppendExecutionProviderCUDA(cudaOpts)
	if err != nil {
		fmt.Println("CUDA EP 初始化失败:", err)
		return
	}

	// 使用 DynamicAdvancedSession
	session, err := ort.NewDynamicAdvancedSession(
		"yolo12x.onnx",
		[]string{"images"},  // 输入节点名称
		[]string{"output0"}, // 输出节点名称
		opts,
	)
	if err != nil {
		panic(fmt.Sprintf("创建 Session 失败: %v", err))
	}
	defer session.Destroy()

	fmt.Println("✅ CUDA 初始化成功，已启用 GPU 推理")

	// 创建输入张量进行推理
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputData := make([]float32, 1*3*640*640)

	// 正确创建输入张量
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		fmt.Printf("创建输入张量失败: %v\n", err)
		return
	}
	defer inputTensor.Destroy()

	// 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400) // YOLO 标准输出形状
	outputData := make([]float32, 1*84*8400)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		fmt.Printf("创建输出张量失败: %v\n", err)
		return
	}
	defer outputTensor.Destroy()

	// 运行推理 - 正确的 API 调用
	err = session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		fmt.Printf("推理失败: %v\n", err)
		return
	}

	fmt.Println("✅ 推理成功")
	fmt.Printf("输出张量形状: %v\n", outputTensor.GetShape())

	// 对比现有项目的CUDA初始化方式
	fmt.Println("\n📊 成功的CUDA初始化关键点:")
	fmt.Println("1. 先设置库路径: ort.SetSharedLibraryPath()")
	fmt.Println("2. 初始化环境: ort.InitializeEnvironment()")
	fmt.Println("3. 创建SessionOptions: ort.NewSessionOptions()")
	fmt.Println("4. 创建CUDA Provider Options: ort.NewCUDAProviderOptions()")
	fmt.Println("5. 更新CUDA选项: cudaOpts.Update()")
	fmt.Println("6. 添加CUDA执行提供者: opts.AppendExecutionProviderCUDA()")
	fmt.Println("7. 创建Session: ort.NewDynamicAdvancedSession()")
	fmt.Println("\n💡 与现有项目的主要区别:")
	fmt.Println("- 现有项目在GPU初始化失败时会尝试DirectML回退")
	fmt.Println("- 用户的成功案例直接使用CUDA，没有回退机制")
	fmt.Println("- 建议在现有项目中优化CUDA初始化顺序和错误处理")
}