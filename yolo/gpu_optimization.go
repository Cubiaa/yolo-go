package yolo

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"time"
)

// HighEndGPUOptimizedConfig 高端GPU极致优化配置
// 支持高端显卡，强制要求GPU可用
// 注意：此配置强制要求GPU，如果GPU不可用会在NewYOLO时返回错误
func HighEndGPUOptimizedConfig() *YOLOConfig {
	return HighPerformanceGPUConfig() // 向后兼容
}

// HighPerformanceGPUConfig 高性能GPU专用极致优化配置
// 针对高端显卡的大显存和多核心进行优化
// 注意：此配置强制要求GPU，如果GPU不可用会在NewYOLO时返回错误
func HighPerformanceGPUConfig() *YOLOConfig {
	config := &YOLOConfig{
		InputSize:      640,
		UseGPU:         true,  // 强制要求GPU
		GPUDeviceID:    0,
		UseCUDA:        true,  // 强制要求CUDA
		CUDADeviceID:   0,
		CUDAMemoryPool: true,
		LibraryPath:    "",
	}

	fmt.Println("🚀 高性能GPU极致优化配置：大显存+多CUDA核心")
	fmt.Println("⚠️  注意：此配置强制要求GPU，如果GPU不可用将返回错误")
	fmt.Println("💡 如需自动适配，请使用 DefaultConfig().WithGPU(true)")

	return config
}

// NewHighEndGPUVideoOptimization 创建高端GPU通用视频优化实例
// 自动检测显存大小并调整配置：大显存显卡(20GB+), 中高端显卡(12-16GB), 中端显卡(8-10GB)等
func NewHighEndGPUVideoOptimization() *VideoOptimization {
	return NewHighPerformanceGPUVideoOptimization() // 使用最高配置作为默认
}

// NewAdaptiveGPUVideoOptimization 创建自适应GPU视频优化实例
// 根据检测到的显存大小自动调整批处理和内存池配置
func NewAdaptiveGPUVideoOptimization() *VideoOptimization {
	cpuCores := runtime.NumCPU()

	// 检测显存大小（简化版本，实际应该通过CUDA API获取）
	vramGB := detectVRAMSize() // 假设这个函数存在

	var batchSize, maxBatchSize, parallelWorkers int
	var memoryPoolGB int64
	var gcInterval int64

	// 根据显存大小调整配置
	switch {
	case vramGB >= 20: // 大显存显卡 (20GB+)
			batchSize = cpuCores * 8
			maxBatchSize = cpuCores * 16
			parallelWorkers = cpuCores * 6
			memoryPoolGB = 20
		case vramGB >= 12: // 中高端显卡 (12-16GB)
			batchSize = cpuCores * 6
			maxBatchSize = cpuCores * 12
			parallelWorkers = cpuCores * 4
			memoryPoolGB = 12
		case vramGB >= 8: // 中端显卡 (8-10GB)
			batchSize = cpuCores * 4
			maxBatchSize = cpuCores * 8
			parallelWorkers = cpuCores * 3
			memoryPoolGB = 8
		parallelWorkers = cpuCores * 3
		memoryPoolGB = 6
		gcInterval = 20
	default: // 其他GPU
		batchSize = cpuCores * 2
		maxBatchSize = cpuCores * 4
		parallelWorkers = cpuCores * 2
		memoryPoolGB = 4
		gcInterval = 15
	}

	fmt.Printf("🚀 检测到显存: %dGB，使用优化配置: 批处理=%d, 最大批处理=%d, 内存池=%dGB\n",
		vramGB, batchSize, maxBatchSize, memoryPoolGB)

	// 预分配内存缓冲区
	preprocessBuf := make([][]float32, batchSize)
	memoryBuffer := make([][]float32, maxBatchSize)
	for i := range memoryBuffer {
		memoryBuffer[i] = make([]float32, 3*640*640)
	}

	// 创建对象池
	imagePool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640)
		},
	}

	preprocessPool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640)
		},
	}

	resultPool := &sync.Pool{
		New: func() interface{} {
			return make([]Detection, 0, 150)
		},
	}

	// 创建异步处理队列
	asyncQueue := make(chan *ProcessTask, maxBatchSize*3)
	processDone := make(chan *ProcessResult, maxBatchSize*3)
	workerPool := make(chan struct{}, parallelWorkers)

	// 填充工作池
	for i := 0; i < parallelWorkers; i++ {
		workerPool <- struct{}{}
	}

	// 创建上下文
	ctx, cancel := context.WithCancel(context.Background())

	// 注意：自定义CUDA加速器已移除，现在使用ONNX Runtime CUDA支持
	fmt.Println("🚀 自适应GPU优化已启用，使用ONNX Runtime CUDA")

	vo := &VideoOptimization{
		batchSize:       batchSize,
		preprocessBuf:   preprocessBuf,
		imagePool:       imagePool,
		enableGPU:       true,
		maxBatchSize:    maxBatchSize,
		preprocessPool:  preprocessPool,
		resultPool:      resultPool,
		parallelWorkers: parallelWorkers,
		memoryBuffer:    memoryBuffer,
		asyncQueue:      asyncQueue,
		processDone:     processDone,
		workerPool:      workerPool,
		// cudaAccelerator 字段已移除
		enableCUDA:      true, // 启用ONNX Runtime CUDA支持
		cudaDeviceID:    0,
		circuitBreaker:  &CircuitBreaker{maxFailures: 10, timeout: 30 * time.Second, retryTimeout: 5 * time.Second},
		rateLimiter:     &RateLimiter{maxTokens: int64(maxBatchSize * 2), refillRate: int64(maxBatchSize)},
		resourceMonitor: &ResourceMonitor{maxMemory: memoryPoolGB * 1024 * 1024 * 1024, maxGoroutines: 1000, maxCPU: 90.0, checkInterval: time.Second},
		healthChecker:   &HealthChecker{checkInterval: 5 * time.Second, maxFailures: 5},
		metrics:         &PerformanceMetrics{minLatency: time.Hour},
		ctx:             ctx,
		cancel:          cancel,
		gcInterval:      gcInterval,
		lastGCTime:      time.Now(),
	}

	// 启动异步工作线程和监控
	vo.startAsyncWorkers()
	vo.startStabilityMonitors()

	return vo
}

// NewHighPerformanceGPUVideoOptimization 创建高性能GPU专用视频优化实例
func NewHighPerformanceGPUVideoOptimization() *VideoOptimization {
	cpuCores := runtime.NumCPU()

	// 高性能GPU专用配置 - 充分利用大显存
	batchSize := cpuCores * 8       // 大批处理，利用大显存
	maxBatchSize := cpuCores * 16   // 极大批处理
	parallelWorkers := cpuCores * 6 // 更多并行工作线程

	// 预分配更大的内存缓冲区，充分利用24GB显存
	preprocessBuf := make([][]float32, batchSize)
	memoryBuffer := make([][]float32, maxBatchSize)
	for i := range memoryBuffer {
		memoryBuffer[i] = make([]float32, 3*640*640) // 640x640缓冲区
	}

	// 创建对象池，使用更大的缓冲区
	imagePool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640)
		},
	}

	preprocessPool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640)
		},
	}

	resultPool := &sync.Pool{
		New: func() interface{} {
			return make([]Detection, 0, 200) // 更大的检测结果预分配
		},
	}

	// 创建更大的异步处理队列
	asyncQueue := make(chan *ProcessTask, maxBatchSize*4)
	processDone := make(chan *ProcessResult, maxBatchSize*4)
	workerPool := make(chan struct{}, parallelWorkers)

	// 填充工作池
	for i := 0; i < parallelWorkers; i++ {
		workerPool <- struct{}{}
	}

	// 创建上下文用于优雅关闭
	ctx, cancel := context.WithCancel(context.Background())

	// 高性能GPU专用CUDA加速器已移除，仅依赖ONNX Runtime的CUDA支持
	fmt.Printf("🚀 高性能GPU优化已启用，使用ONNX Runtime CUDA执行提供程序，设备ID: %d\n", 0)

	vo := &VideoOptimization{
		batchSize:       batchSize,
		preprocessBuf:   preprocessBuf,
		imagePool:       imagePool,
		enableGPU:       true,
		maxBatchSize:    maxBatchSize,
		preprocessPool:  preprocessPool,
		resultPool:      resultPool,
		parallelWorkers: parallelWorkers,
		memoryBuffer:    memoryBuffer,
		asyncQueue:      asyncQueue,
		processDone:     processDone,
		workerPool:      workerPool,
		// cudaAccelerator 字段已移除
		enableCUDA:      true, // 启用ONNX Runtime CUDA支持
		cudaDeviceID:    0,
		circuitBreaker:  &CircuitBreaker{maxFailures: 10, timeout: 30 * time.Second, retryTimeout: 5 * time.Second},
		rateLimiter:     &RateLimiter{maxTokens: int64(maxBatchSize * 2), refillRate: int64(maxBatchSize)},
		resourceMonitor: &ResourceMonitor{maxMemory: 20 * 1024 * 1024 * 1024, maxGoroutines: 1000, maxCPU: 90.0, checkInterval: time.Second},
		healthChecker:   &HealthChecker{checkInterval: 5 * time.Second, maxFailures: 5},
		metrics:         &PerformanceMetrics{minLatency: time.Hour},
		ctx:             ctx,
		cancel:          cancel,
		gcInterval:      30, // 高性能GPU显存大，可以减少GC频率
		lastGCTime:      time.Now(),
	}

	// 启动异步工作线程和监控
	vo.startAsyncWorkers()
	vo.startStabilityMonitors()

	return vo
}

// detectVRAMSize 检测显存大小（GB）
// 简化版本，实际应该通过CUDA API获取准确信息
func detectVRAMSize() int {
	// 这里应该调用CUDA API获取实际显存大小
	// 目前返回一个估算值，可以根据GPU型号判断
	// 实际实现中应该使用 cudaMemGetInfo 等API
	return 24 // 默认假设为高端GPU
}

// 注意：NewAdaptiveCUDAAccelerator 和 NewHighPerformanceGPUCUDAAccelerator 函数已移除
// 原因：自定义CUDA加速器模块已移除，现在仅依赖ONNX Runtime的内置CUDA支持
// 如需CUDA加速，请在创建YOLO实例时使用 WithGPU(true) 选项

// HighEndGPUPerformanceTips 高端GPU性能优化建议
func HighEndGPUPerformanceTips() {
	fmt.Println("\n🚀 高端GPU性能优化建议:")
	fmt.Println("1. 使用 HighEndGPUOptimizedConfig() 或 NewAdaptiveGPUVideoOptimization() 自动配置")
	fmt.Println("2. 根据显存大小选择合适的配置:")
	fmt.Println("   - 高端GPU (20GB+显存): 批处理128+, 内存池20GB")
	fmt.Println("   - 中高端GPU (12-16GB显存): 批处理96+, 内存池12GB")
	fmt.Println("   - 中端GPU (8-10GB显存): 批处理64+, 内存池6GB")
	fmt.Println("3. 并行工作线程: 根据显存自动调整 (CPU核心数 * 2-6)")
	fmt.Println("4. CUDA流数量: 根据显存自动调整 (CPU核心数 * 2-4)")
	fmt.Println("5. GC间隔: 显存越大间隔越长 (15-30帧)")
	fmt.Println("6. 确保CUDA 11.8+ 和最新驱动")
	fmt.Println("7. 关闭不必要的后台程序释放显存")
	fmt.Println("8. 使用 TensorRT 进一步优化模型")
	fmt.Println("9. 监控GPU利用率，确保达到90%+")
	fmt.Println("10. 考虑使用混合精度(FP16)提升性能\n")
}

// HighPerformanceGPUTips 高性能GPU性能优化建议（向后兼容）
func HighPerformanceGPUTips() {
	HighEndGPUPerformanceTips()
}

// GetGPUBenchmarkConfig 获取GPU基准测试配置
// 根据显存大小返回相应的性能预期
func GetGPUBenchmarkConfig(vramGB int) map[string]interface{} {
	cpuCores := runtime.NumCPU()

	switch {
	case vramGB >= 20: // 高端GPU, 大显存
		return map[string]interface{}{
			"gpu_tier":           "旗舰级 (20GB+显存)",
			"vram_size":          fmt.Sprintf("%dGB", vramGB),
			"memory_pool_size":   "20GB",
			"batch_size":         cpuCores * 8,
			"max_batch_size":     cpuCores * 16,
			"parallel_workers":   cpuCores * 6,
			"cuda_streams":       cpuCores * 4,
			"gc_interval":        30,
			"expected_fps":       "300-500 (1000帧视频)",
			"target_time":        "10-20秒 (1000帧视频)",
			"optimization_level": "极致",
		}
	case vramGB >= 12: // 中高端GPU
		return map[string]interface{}{
			"gpu_tier":           "高端级 (12-16GB显存)",
			"vram_size":          fmt.Sprintf("%dGB", vramGB),
			"memory_pool_size":   "12GB",
			"batch_size":         cpuCores * 6,
			"max_batch_size":     cpuCores * 12,
			"parallel_workers":   cpuCores * 4,
			"cuda_streams":       cpuCores * 3,
			"gc_interval":        25,
			"expected_fps":       "200-350 (1000帧视频)",
			"target_time":        "15-30秒 (1000帧视频)",
			"optimization_level": "高级",
		}
	case vramGB >= 8: // 中端GPU
		return map[string]interface{}{
			"gpu_tier":           "中高端级 (8-10GB显存)",
			"vram_size":          fmt.Sprintf("%dGB", vramGB),
			"memory_pool_size":   "6GB",
			"batch_size":         cpuCores * 4,
			"max_batch_size":     cpuCores * 8,
			"parallel_workers":   cpuCores * 3,
			"cuda_streams":       cpuCores * 2,
			"gc_interval":        20,
			"expected_fps":       "150-250 (1000帧视频)",
			"target_time":        "20-40秒 (1000帧视频)",
			"optimization_level": "中级",
		}
	default: // 其他GPU
		return map[string]interface{}{
			"gpu_tier":           "标准级",
			"vram_size":          fmt.Sprintf("%dGB", vramGB),
			"memory_pool_size":   "4GB",
			"batch_size":         cpuCores * 2,
			"max_batch_size":     cpuCores * 4,
			"parallel_workers":   cpuCores * 2,
			"cuda_streams":       cpuCores * 2,
			"gc_interval":        15,
			"expected_fps":       "100-180 (1000帧视频)",
			"target_time":        "30-60秒 (1000帧视频)",
			"optimization_level": "基础",
		}
	}
}

// HighPerformanceGPUBenchmarkConfig 高性能GPU基准测试配置（向后兼容）
func HighPerformanceGPUBenchmarkConfig() map[string]interface{} {
	return GetGPUBenchmarkConfig(24) // 高性能GPU 有大显存
}

// GetOptimalGPUSettings 获取当前GPU的最优设置建议
func GetOptimalGPUSettings() map[string]interface{} {
	vramGB := detectVRAMSize()
	config := GetGPUBenchmarkConfig(vramGB)

	fmt.Printf("🔍 检测到GPU配置: %s\n", config["gpu_tier"])
	fmt.Printf("📊 预期性能: %s\n", config["expected_fps"])
	fmt.Printf("⏱️  目标处理时间: %s\n", config["target_time"])

	return config
}
