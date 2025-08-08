package yolo

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
)

// ImprovedCUDAInitializer 改进的CUDA初始化器
// 基于用户成功案例的CUDA初始化方式
type ImprovedCUDAInitializer struct {
	libraryPath string
	deviceID    int
	initialized bool
}

// NewImprovedCUDAInitializer 创建改进的CUDA初始化器
func NewImprovedCUDAInitializer(libraryPath string, deviceID int) *ImprovedCUDAInitializer {
	return &ImprovedCUDAInitializer{
		libraryPath: libraryPath,
		deviceID:    deviceID,
		initialized: false,
	}
}

// InitializeCUDAWithSuccessfulMethod 使用成功的方法初始化CUDA
// 基于用户提供的成功案例
func (ici *ImprovedCUDAInitializer) InitializeCUDAWithSuccessfulMethod() (*ort.SessionOptions, error) {
	fmt.Println("🚀 使用改进的CUDA初始化方法")

	// 步骤1: 设置 ONNX Runtime 库路径
	if ici.libraryPath != "" {
		ort.SetSharedLibraryPath(ici.libraryPath)
		fmt.Printf("📁 设置库路径: %s\n", ici.libraryPath)
	}

	// 步骤2: 初始化环境（如果还未初始化）
	if !ici.initialized {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, fmt.Errorf("初始化ONNX Runtime环境失败: %v", err)
		}
		ici.initialized = true
		fmt.Println("✅ ONNX Runtime环境初始化成功")
	}

	// 步骤3: 创建 SessionOptions
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建SessionOptions失败: %v", err)
	}
	fmt.Println("✅ SessionOptions创建成功")

	// 步骤4: 配置 CUDA Provider
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("创建CUDA Provider Options失败: %v", err)
	}
	defer cudaOpts.Destroy()
	fmt.Println("✅ CUDA Provider Options创建成功")

	// 步骤5: 更新CUDA选项
	err = cudaOpts.Update(map[string]string{
		"device_id": fmt.Sprintf("%d", ici.deviceID),
	})
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("更新CUDA选项失败: %v", err)
	}
	fmt.Printf("✅ CUDA选项更新成功 (设备ID: %d)\n", ici.deviceID)

	// 步骤6: 添加CUDA执行提供者
	err = opts.AppendExecutionProviderCUDA(cudaOpts)
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("添加CUDA执行提供者失败: %v", err)
	}
	fmt.Println("✅ CUDA执行提供者添加成功")

	return opts, nil
}

// CreateSessionWithImprovedCUDA 使用改进的CUDA方法创建Session
func (ici *ImprovedCUDAInitializer) CreateSessionWithImprovedCUDA(modelPath string, inputNames, outputNames []string) (*ort.DynamicAdvancedSession, error) {
	// 使用改进的CUDA初始化方法
	opts, err := ici.InitializeCUDAWithSuccessfulMethod()
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	// 创建Session
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		opts,
	)
	if err != nil {
		return nil, fmt.Errorf("创建Session失败: %v", err)
	}

	fmt.Println("✅ CUDA Session创建成功")
	return session, nil
}

// TestCUDAInference 测试CUDA推理
func (ici *ImprovedCUDAInitializer) TestCUDAInference(session *ort.DynamicAdvancedSession) error {
	fmt.Println("🧪 开始CUDA推理测试")

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputData := make([]float32, 1*3*640*640)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return fmt.Errorf("创建输入张量失败: %v", err)
	}
	defer inputTensor.Destroy()

	// 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400)
	outputData := make([]float32, 1*84*8400)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return fmt.Errorf("创建输出张量失败: %v", err)
	}
	defer outputTensor.Destroy()

	// 运行推理
	err = session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return fmt.Errorf("推理失败: %v", err)
	}

	fmt.Println("✅ CUDA推理测试成功")
	fmt.Printf("📊 输出张量形状: %v\n", outputTensor.GetShape())
	return nil
}

// Cleanup 清理资源
func (ici *ImprovedCUDAInitializer) Cleanup() {
	if ici.initialized {
		ort.DestroyEnvironment()
		ici.initialized = false
		fmt.Println("🧹 ONNX Runtime环境已清理")
	}
}

// GetInitializationSteps 获取初始化步骤说明
func (ici *ImprovedCUDAInitializer) GetInitializationSteps() []string {
	return []string{
		"1. 设置ONNX Runtime库路径: ort.SetSharedLibraryPath()",
		"2. 初始化环境: ort.InitializeEnvironment()",
		"3. 创建SessionOptions: ort.NewSessionOptions()",
		"4. 创建CUDA Provider Options: ort.NewCUDAProviderOptions()",
		"5. 更新CUDA选项: cudaOpts.Update()",
		"6. 添加CUDA执行提供者: opts.AppendExecutionProviderCUDA()",
		"7. 创建Session: ort.NewDynamicAdvancedSession()",
	}
}

// CompareWithCurrentImplementation 与当前实现的对比
func (ici *ImprovedCUDAInitializer) CompareWithCurrentImplementation() map[string]string {
	return map[string]string{
		"当前实现": "在GPU初始化失败时会尝试DirectML回退",
		"改进方案": "直接使用CUDA，提供更清晰的错误信息",
		"优势1":   "初始化步骤更明确，易于调试",
		"优势2":   "减少不必要的回退机制，提高性能",
		"优势3":   "基于用户成功案例，可靠性更高",
		"建议":    "可以作为现有项目的CUDA初始化优化参考",
	}
}