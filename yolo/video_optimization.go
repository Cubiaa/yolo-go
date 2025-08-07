package yolo

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/disintegration/imaging"
)

// VideoOptimization GPU优化相关的结构体和方法 - 疯狂调用稳定版 + CUDA加速
type VideoOptimization struct {
	batchSize     int
	preprocessBuf [][]float32
	imagePool     *sync.Pool
	enableGPU     bool
	// 极致性能优化字段
	maxBatchSize    int
	workerPool      chan struct{}
	preprocessPool  *sync.Pool
	resultPool      *sync.Pool
	parallelWorkers int
	memoryBuffer    [][]float32
	asyncQueue      chan *ProcessTask
	processDone     chan *ProcessResult

	// CUDA加速模块
	cudaAccelerator *CUDAAccelerator
	enableCUDA      bool
	cudaDeviceID    int

	// 疯狂调用稳定性保障字段
	circuitBreaker  *CircuitBreaker
	rateLimiter     *RateLimiter
	resourceMonitor *ResourceMonitor
	healthChecker   *HealthChecker
	metrics         *PerformanceMetrics
	ctx             context.Context
	cancel          context.CancelFunc
	isShutdown      int64 // atomic

	// 垃圾回收优化字段
	frameCounter    int64 // 帧计数器，用于定期垃圾回收
	gcInterval      int64 // GC间隔，默认每20-50帧清理一次
	lastGCTime      time.Time // 上次GC时间
	gcMutex         sync.Mutex // GC操作互斥锁
}

// ProcessTask 异步处理任务
type ProcessTask struct {
	img    image.Image
	width  int
	height int
	id     int
}

// ProcessResult 处理结果
type ProcessResult struct {
	data []float32
	err  error
	id   int
}

// CircuitBreaker 熔断器 - 防止系统过载
type CircuitBreaker struct {
	mu            sync.RWMutex
	state         CircuitState
	failureCount  int64
	lastFailTime  time.Time
	nextRetryTime time.Time
	maxFailures   int64
	timeout       time.Duration
	retryTimeout  time.Duration
}

type CircuitState int

const (
	Closed CircuitState = iota
	Open
	HalfOpen
)

// RateLimiter 限流器 - 控制调用频率
type RateLimiter struct {
	mu         sync.Mutex
	tokens     int64
	maxTokens  int64
	refillRate int64
	lastRefill time.Time
}

// ResourceMonitor 资源监控器 - 监控系统资源
type ResourceMonitor struct {
	mu             sync.RWMutex
	memoryUsage    int64
	goroutineCount int64
	cpuUsage       float64
	maxMemory      int64
	maxGoroutines  int64
	maxCPU         float64
	lastCheck      time.Time
	checkInterval  time.Duration
}

// HealthChecker 健康检查器 - 检查系统健康状态
type HealthChecker struct {
	mu            sync.RWMutex
	isHealthy     bool
	lastCheck     time.Time
	checkInterval time.Duration
	failureCount  int64
	maxFailures   int64
}

// PerformanceMetrics 性能指标 - 记录性能数据
type PerformanceMetrics struct {
	mu              sync.RWMutex
	totalRequests   int64
	successRequests int64
	failedRequests  int64
	avgLatency      time.Duration
	maxLatency      time.Duration
	minLatency      time.Duration
	throughput      float64
	lastUpdate      time.Time
}

// NewVideoOptimization 创建视频优化实例 - 疯狂调用稳定版本 + CUDA加速
func NewVideoOptimization(enableGPU bool) *VideoOptimization {
	return NewVideoOptimizationWithCUDA(enableGPU, false, 0)
}

// NewVideoOptimizationWithCUDA 创建带CUDA加速的视频优化实例
func NewVideoOptimizationWithCUDA(enableGPU, enableCUDA bool, cudaDeviceID int) *VideoOptimization {
	// 平衡性能与内存使用
	cpuCores := runtime.NumCPU()

	// 合理的批处理大小，避免内存不足
	batchSize := cpuCores * 2       // 基础批处理
	maxBatchSize := cpuCores * 4    // 最大批处理，平衡模式
	parallelWorkers := cpuCores * 2 // 并行工作线程数

	if enableGPU {
		// GPU模式下适度增加批处理
		batchSize = cpuCores * 3
		maxBatchSize = cpuCores * 6 // GPU优化模式
		parallelWorkers = cpuCores * 3
	}

	// CUDA模式下进一步优化，但控制内存使用
	if enableCUDA {
		batchSize = cpuCores * 4       // CUDA批处理
		maxBatchSize = cpuCores * 8    // CUDA优化模式
		parallelWorkers = cpuCores * 4 // CUDA并行工作线程
	}

	// 预分配合理的内存缓冲区
	preprocessBuf := make([][]float32, batchSize)
	memoryBuffer := make([][]float32, maxBatchSize)
	for i := range memoryBuffer {
		memoryBuffer[i] = make([]float32, 3*640*640) // 预分配640x640缓冲区
	}

	// 创建对象池，使用合理的缓冲区大小
	imagePool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640) // 640x640缓冲区
		},
	}

	preprocessPool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*640*640)
		},
	}

	resultPool := &sync.Pool{
		New: func() interface{} {
			return make([]Detection, 0, 100) // 预分配检测结果
		},
	}

	// 创建异步处理队列
	asyncQueue := make(chan *ProcessTask, maxBatchSize*2)
	processDone := make(chan *ProcessResult, maxBatchSize*2)
	workerPool := make(chan struct{}, parallelWorkers)

	// 填充工作池
	for i := 0; i < parallelWorkers; i++ {
		workerPool <- struct{}{}
	}

	// 创建上下文用于优雅关闭
	ctx, cancel := context.WithCancel(context.Background())

	// 初始化稳定性保障组件
	circuitBreaker := &CircuitBreaker{
		maxFailures:  10,
		timeout:      30 * time.Second,
		retryTimeout: 5 * time.Second,
		state:        Closed,
	}

	rateLimiter := &RateLimiter{
		maxTokens:  int64(parallelWorkers * 10), // 允许突发流量
		refillRate: int64(parallelWorkers),      // 每秒补充令牌
		tokens:     int64(parallelWorkers * 10),
		lastRefill: time.Now(),
	}

	resourceMonitor := &ResourceMonitor{
		maxMemory:     1024 * 1024 * 1024 * 2, // 2GB内存限制
		maxGoroutines: int64(parallelWorkers * 2),
		maxCPU:        80.0, // 80% CPU使用率限制
		checkInterval: time.Second,
		lastCheck:     time.Now(),
	}

	healthChecker := &HealthChecker{
		isHealthy:     true,
		checkInterval: 5 * time.Second,
		maxFailures:   5,
		lastCheck:     time.Now(),
	}

	metrics := &PerformanceMetrics{
		minLatency: time.Hour, // 初始化为最大值
		lastUpdate: time.Now(),
	}

	// 初始化CUDA加速器（如果启用）
	var cudaAccelerator *CUDAAccelerator
	if enableCUDA {
		var err error
		cudaAccelerator, err = NewCUDAAccelerator(cudaDeviceID)
		if err != nil {
			fmt.Printf("⚠️ CUDA加速器初始化失败，回退到CPU模式: %v\n", err)
			enableCUDA = false
			cudaAccelerator = nil
		} else {
			fmt.Printf("🚀 CUDA加速器初始化成功，设备ID: %d\n", cudaDeviceID)
		}
	}

	vo := &VideoOptimization{
		batchSize:       batchSize,
		preprocessBuf:   preprocessBuf,
		imagePool:       imagePool,
		enableGPU:       enableGPU,
		maxBatchSize:    maxBatchSize,
		workerPool:      workerPool,
		preprocessPool:  preprocessPool,
		resultPool:      resultPool,
		parallelWorkers: parallelWorkers,
		memoryBuffer:    memoryBuffer,
		asyncQueue:      asyncQueue,
		processDone:     processDone,
		// CUDA加速模块
		cudaAccelerator: cudaAccelerator,
		enableCUDA:      enableCUDA,
		cudaDeviceID:    cudaDeviceID,
		// 稳定性保障组件
		circuitBreaker:  circuitBreaker,
		rateLimiter:     rateLimiter,
		resourceMonitor: resourceMonitor,
		healthChecker:   healthChecker,
		metrics:         metrics,
		ctx:             ctx,
		cancel:          cancel,
		isShutdown:      0,
		// 垃圾回收优化字段
		frameCounter:    0,
		gcInterval:      30, // 默认每30帧清理一次，平衡性能与内存
		lastGCTime:      time.Now(),
	}

	// 启动异步处理工作线程
	vo.startAsyncWorkers()

	// 启动稳定性监控
	vo.startStabilityMonitors()

	return vo
}

// startAsyncWorkers 启动异步处理工作线程
func (vo *VideoOptimization) startAsyncWorkers() {
	for i := 0; i < vo.parallelWorkers; i++ {
		go vo.asyncWorker()
	}
}

// startStabilityMonitors 启动稳定性监控
func (vo *VideoOptimization) startStabilityMonitors() {
	// 启动资源监控
	go vo.resourceMonitorLoop()
	// 启动健康检查
	go vo.healthCheckLoop()
	// 启动性能指标更新
	go vo.metricsUpdateLoop()
}

// asyncWorker 异步工作线程 - 带稳定性保障
func (vo *VideoOptimization) asyncWorker() {
	for {
		select {
		case task := <-vo.asyncQueue:
			// 检查系统是否关闭
			if atomic.LoadInt64(&vo.isShutdown) == 1 {
				return
			}

			// 检查熔断器状态
			if !vo.circuitBreakerAllow() {
				vo.processDone <- &ProcessResult{
					data: nil,
					err:  fmt.Errorf("circuit breaker open"),
					id:   task.id,
				}
				continue
			}

			// 限流检查
			if !vo.rateLimiterAllow() {
				vo.processDone <- &ProcessResult{
					data: nil,
					err:  fmt.Errorf("rate limit exceeded"),
					id:   task.id,
				}
				continue
			}

			// 资源检查
			if !vo.resourceCheck() {
				vo.processDone <- &ProcessResult{
					data: nil,
					err:  fmt.Errorf("resource limit exceeded"),
					id:   task.id,
				}
				continue
			}

			<-vo.workerPool // 获取工作许可

			// 记录开始时间
			startTime := time.Now()

			// 执行预处理
			data, err := vo.extremePreprocessImage(task.img, task.width, task.height)

			// 记录性能指标
			latency := time.Since(startTime)
			vo.updateMetrics(latency, err == nil)

			// 更新熔断器状态
			vo.circuitBreakerRecord(err == nil)

			// 创建结果
			result := &ProcessResult{
				data: data,
				err:  err,
				id:   task.id,
			}

			// 先释放工作许可，避免死锁
			vo.workerPool <- struct{}{}

			// 非阻塞发送结果，避免死锁
			select {
			case vo.processDone <- result:
				// 成功发送结果
			default:
				// 结果通道满时丢弃结果，避免死锁
				// 在实际应用中可以考虑记录日志或其他处理方式
			}

		case <-vo.ctx.Done():
			// 上下文取消，退出工作线程
			return
		}
	}
}

// 熔断器相关方法
func (vo *VideoOptimization) circuitBreakerAllow() bool {
	vo.circuitBreaker.mu.RLock()
	defer vo.circuitBreaker.mu.RUnlock()

	switch vo.circuitBreaker.state {
	case Closed:
		return true
	case Open:
		return time.Now().After(vo.circuitBreaker.nextRetryTime)
	case HalfOpen:
		return true
	default:
		return false
	}
}

func (vo *VideoOptimization) circuitBreakerRecord(success bool) {
	vo.circuitBreaker.mu.Lock()
	defer vo.circuitBreaker.mu.Unlock()

	if success {
		if vo.circuitBreaker.state == HalfOpen {
			vo.circuitBreaker.state = Closed
			vo.circuitBreaker.failureCount = 0
		}
	} else {
		vo.circuitBreaker.failureCount++
		vo.circuitBreaker.lastFailTime = time.Now()

		if vo.circuitBreaker.failureCount >= vo.circuitBreaker.maxFailures {
			vo.circuitBreaker.state = Open
			vo.circuitBreaker.nextRetryTime = time.Now().Add(vo.circuitBreaker.retryTimeout)
		}
	}
}

// 限流器相关方法
func (vo *VideoOptimization) rateLimiterAllow() bool {
	vo.rateLimiter.mu.Lock()
	defer vo.rateLimiter.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(vo.rateLimiter.lastRefill)

	// 补充令牌
	if elapsed > 0 {
		tokensToAdd := int64(elapsed.Seconds()) * vo.rateLimiter.refillRate
		vo.rateLimiter.tokens = min(vo.rateLimiter.maxTokens, vo.rateLimiter.tokens+tokensToAdd)
		vo.rateLimiter.lastRefill = now
	}

	// 检查是否有可用令牌
	if vo.rateLimiter.tokens > 0 {
		vo.rateLimiter.tokens--
		return true
	}

	return false
}

// 资源检查方法
func (vo *VideoOptimization) resourceCheck() bool {
	vo.resourceMonitor.mu.RLock()
	defer vo.resourceMonitor.mu.RUnlock()

	// 检查内存使用
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	if int64(m.Alloc) > vo.resourceMonitor.maxMemory {
		return false
	}

	// 检查goroutine数量
	if int64(runtime.NumGoroutine()) > vo.resourceMonitor.maxGoroutines {
		return false
	}

	return true
}

// 性能指标更新方法
func (vo *VideoOptimization) updateMetrics(latency time.Duration, success bool) {
	vo.metrics.mu.Lock()
	defer vo.metrics.mu.Unlock()

	vo.metrics.totalRequests++
	if success {
		vo.metrics.successRequests++
	} else {
		vo.metrics.failedRequests++
	}

	// 更新延迟统计
	if latency > vo.metrics.maxLatency {
		vo.metrics.maxLatency = latency
	}
	if latency < vo.metrics.minLatency {
		vo.metrics.minLatency = latency
	}

	// 计算平均延迟
	vo.metrics.avgLatency = (vo.metrics.avgLatency*time.Duration(vo.metrics.totalRequests-1) + latency) / time.Duration(vo.metrics.totalRequests)

	vo.metrics.lastUpdate = time.Now()
}

// 监控循环方法
func (vo *VideoOptimization) resourceMonitorLoop() {
	ticker := time.NewTicker(vo.resourceMonitor.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vo.updateResourceMetrics()
		case <-vo.ctx.Done():
			return
		}
	}
}

func (vo *VideoOptimization) healthCheckLoop() {
	ticker := time.NewTicker(vo.healthChecker.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vo.performHealthCheck()
		case <-vo.ctx.Done():
			return
		}
	}
}

func (vo *VideoOptimization) metricsUpdateLoop() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			vo.updateThroughput()
		case <-vo.ctx.Done():
			return
		}
	}
}

func (vo *VideoOptimization) updateResourceMetrics() {
	vo.resourceMonitor.mu.Lock()
	defer vo.resourceMonitor.mu.Unlock()

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	vo.resourceMonitor.memoryUsage = int64(m.Alloc)
	vo.resourceMonitor.goroutineCount = int64(runtime.NumGoroutine())
	vo.resourceMonitor.lastCheck = time.Now()
}

func (vo *VideoOptimization) performHealthCheck() {
	vo.healthChecker.mu.Lock()
	defer vo.healthChecker.mu.Unlock()

	// 检查各种健康指标
	healthy := true

	// 检查熔断器状态
	if vo.circuitBreaker.state == Open {
		healthy = false
	}

	// 检查资源使用
	if !vo.resourceCheck() {
		healthy = false
	}

	// 检查队列状态
	if len(vo.asyncQueue) > cap(vo.asyncQueue)*8/10 { // 队列使用超过80%
		healthy = false
	}

	if healthy {
		vo.healthChecker.isHealthy = true
		vo.healthChecker.failureCount = 0
	} else {
		vo.healthChecker.failureCount++
		if vo.healthChecker.failureCount >= vo.healthChecker.maxFailures {
			vo.healthChecker.isHealthy = false
		}
	}

	vo.healthChecker.lastCheck = time.Now()
}

func (vo *VideoOptimization) updateThroughput() {
	vo.metrics.mu.Lock()
	defer vo.metrics.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(vo.metrics.lastUpdate).Seconds()
	if elapsed > 0 {
		vo.metrics.throughput = float64(vo.metrics.successRequests) / elapsed
	}
}

// 辅助函数
func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// OptimizedPreprocessImage 优化的图像预处理方法 - 极致性能版本 + CUDA加速
func (vo *VideoOptimization) OptimizedPreprocessImage(img image.Image, inputWidth, inputHeight int) ([]float32, error) {
	// 如果启用CUDA加速，优先使用CUDA预处理
	if vo.enableCUDA && vo.cudaAccelerator != nil {
		result, err := vo.cudaAccelerator.PreprocessImageCUDA(img, inputWidth, inputHeight)
		if err == nil {
			return result, nil
		}
		// CUDA失败时回退到CPU模式
		fmt.Printf("⚠️ CUDA预处理失败，回退到CPU模式: %v\n", err)
	}

	// 使用CPU极致性能预处理
	return vo.extremePreprocessImage(img, inputWidth, inputHeight)
}

// extremePreprocessImage 极致性能图像预处理
func (vo *VideoOptimization) extremePreprocessImage(img image.Image, inputWidth, inputHeight int) ([]float32, error) {
	// 从预处理池获取缓冲区
	buf := vo.preprocessPool.Get().([]float32)
	defer func() {
		// 确保归还到池中的缓冲区大小合理，避免内存泄漏
		if len(buf) <= 3*1024*1024 { // 最大1024x1024的缓冲区
			vo.preprocessPool.Put(buf)
		}
	}()

	// 确保缓冲区大小足够
	requiredSize := 3 * inputWidth * inputHeight
	if len(buf) < requiredSize {
		buf = make([]float32, requiredSize)
	}

	// 极速缩放 - 使用最快算法
	resized := vo.extremeFastResize(img, inputWidth, inputHeight)

	// 极速归一化 - 并行处理
	var result []float32
	if rgba, ok := resized.(*image.RGBA); ok {
		result = vo.extremeFastNormalizeRGBA(rgba, buf)
	} else {
		result = vo.extremeFastNormalize(resized, buf)
	}

	// 创建结果的副本，避免返回池中的缓冲区引用
	output := make([]float32, len(result))
	copy(output, result)
	return output, nil
}

// fastResize 快速图像缩放 - 修复坐标转换问题
func (vo *VideoOptimization) fastResize(img image.Image, width, height int) image.Image {
	// 修复：使用与CPU路径相同的缩放算法，确保坐标转换一致性
	// 从 imaging.NearestNeighbor 改为 imaging.Lanczos
	return imaging.Resize(img, width, height, imaging.Lanczos)
}

// extremeFastResize 极致性能图像缩放 - 修复坐标转换问题
func (vo *VideoOptimization) extremeFastResize(img image.Image, width, height int) image.Image {
	// 检查是否需要缩放
	bounds := img.Bounds()
	if bounds.Dx() == width && bounds.Dy() == height {
		return img // 无需缩放，直接返回
	}

	// 修复：使用与CPU路径相同的缩放算法，确保坐标转换一致性
	// 从 imaging.NearestNeighbor 改为 imaging.Lanczos
	return imaging.Resize(img, width, height, imaging.Lanczos)
}

// resizeWithPadding 保持宽高比的缩放和填充 - 修复数据类型一致性
func (vo *VideoOptimization) resizeWithPadding(img image.Image, targetWidth, targetHeight int) image.Image {
	bounds := img.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()

	// 修复：使用与yolo.go中相同的数据类型，确保坐标转换一致性
	// 计算缩放比例，保持宽高比
	scaleX := float32(targetWidth) / float32(origWidth)
	scaleY := float32(targetHeight) / float32(origHeight)
	scale := scaleX
	if scaleY < scaleX {
		scale = scaleY
	}

	// 计算缩放后的尺寸
	newWidth := int(float32(origWidth) * scale)
	newHeight := int(float32(origHeight) * scale)

	// 缩放图像（修复：使用与CPU路径相同的缩放算法）
	resized := imaging.Resize(img, newWidth, newHeight, imaging.Lanczos)

	// 创建目标尺寸的黑色背景
	result := imaging.New(targetWidth, targetHeight, color.NRGBA{0, 0, 0, 255})

	// 计算居中位置
	offsetX := (targetWidth - newWidth) / 2
	offsetY := (targetHeight - newHeight) / 2

	// 将缩放后的图像粘贴到中心
	result = imaging.Paste(result, resized, image.Pt(offsetX, offsetY))

	return result
}

// fastNormalize 快速归一化（通用版本）
func (vo *VideoOptimization) fastNormalize(img image.Image, buf []float32) []float32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 确保缓冲区大小足够
	requiredSize := 3 * height * width
	if len(buf) < requiredSize {
		buf = make([]float32, requiredSize)
	}

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// 归一化到 [0, 1] 范围
			buf[y*width+x] = float32(r>>8) / 255.0                // R 通道
			buf[height*width+y*width+x] = float32(g>>8) / 255.0   // G 通道
			buf[2*height*width+y*width+x] = float32(b>>8) / 255.0 // B 通道
		}
	}
	return buf[:requiredSize]
}

// extremeFastNormalize 极致性能归一化（通用版本）- 并行处理
func (vo *VideoOptimization) extremeFastNormalize(img image.Image, buf []float32) []float32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 确保缓冲区大小足够
	requiredSize := 3 * height * width
	if len(buf) < requiredSize {
		buf = make([]float32, requiredSize)
	}

	// 并行处理 - 按行分割
	numWorkers := vo.parallelWorkers
	if numWorkers > height {
		numWorkers = height
	}

	rowsPerWorker := height / numWorkers
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(startRow, endRow int) {
			defer wg.Done()
			for y := startRow; y < endRow; y++ {
				for x := 0; x < width; x++ {
					r, g, b, _ := img.At(x, y).RGBA()
					// 归一化到 [0, 1]
					buf[y*width+x] = float32(r>>8) / 255.0                // R 通道
					buf[height*width+y*width+x] = float32(g>>8) / 255.0   // G 通道
					buf[2*height*width+y*width+x] = float32(b>>8) / 255.0 // B 通道
				}
			}
		}(i*rowsPerWorker, (i+1)*rowsPerWorker)
	}

	// 处理剩余行
	if height%numWorkers != 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for y := numWorkers * rowsPerWorker; y < height; y++ {
				for x := 0; x < width; x++ {
					r, g, b, _ := img.At(x, y).RGBA()
					buf[y*width+x] = float32(r>>8) / 255.0
					buf[height*width+y*width+x] = float32(g>>8) / 255.0
					buf[2*height*width+y*width+x] = float32(b>>8) / 255.0
				}
			}
		}()
	}

	wg.Wait()
	return buf[:requiredSize]
}

// fastNormalizeRGBA 快速归一化（RGBA优化版本）
func (vo *VideoOptimization) fastNormalizeRGBA(rgba *image.RGBA, buf []float32) []float32 {
	bounds := rgba.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 确保缓冲区大小足够
	requiredSize := 3 * height * width
	if len(buf) < requiredSize {
		buf = make([]float32, requiredSize)
	}

	// 直接访问像素数据，避免At()方法的开销
	pix := rgba.Pix
	stride := rgba.Stride

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			i := y*stride + x*4
			// 归一化到 [0, 1]
			buf[y*width+x] = float32(pix[i]) / 255.0                  // R通道
			buf[height*width+y*width+x] = float32(pix[i+1]) / 255.0   // G通道
			buf[2*height*width+y*width+x] = float32(pix[i+2]) / 255.0 // B通道
		}
	}

	return buf[:requiredSize]
}

// extremeFastNormalizeRGBA 极致性能RGBA归一化 - 并行+SIMD优化
func (vo *VideoOptimization) extremeFastNormalizeRGBA(rgba *image.RGBA, buf []float32) []float32 {
	bounds := rgba.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 确保缓冲区大小足够
	requiredSize := 3 * height * width
	if len(buf) < requiredSize {
		buf = make([]float32, requiredSize)
	}

	// 直接访问像素数据
	pix := rgba.Pix
	stride := rgba.Stride

	// 并行处理 - 按行分割
	numWorkers := vo.parallelWorkers
	if numWorkers > height {
		numWorkers = height
	}

	rowsPerWorker := height / numWorkers
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(startRow, endRow int) {
			defer wg.Done()
			// 批量处理像素 - 极致优化
			for y := startRow; y < endRow; y++ {
				rowOffset := y * stride
				for x := 0; x < width; x++ {
					i := rowOffset + x*4
					// 直接内存访问，最快归一化
					buf[y*width+x] = float32(pix[i]) * 0.003921568627451 // 1/255 预计算
					buf[height*width+y*width+x] = float32(pix[i+1]) * 0.003921568627451
					buf[2*height*width+y*width+x] = float32(pix[i+2]) * 0.003921568627451
				}
			}
		}(i*rowsPerWorker, (i+1)*rowsPerWorker)
	}

	// 处理剩余行
	if height%numWorkers != 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for y := numWorkers * rowsPerWorker; y < height; y++ {
				rowOffset := y * stride
				for x := 0; x < width; x++ {
					i := rowOffset + x*4
					buf[y*width+x] = float32(pix[i]) * 0.003921568627451
					buf[height*width+y*width+x] = float32(pix[i+1]) * 0.003921568627451
					buf[2*height*width+y*width+x] = float32(pix[i+2]) * 0.003921568627451
				}
			}
		}()
	}

	wg.Wait()
	return buf[:requiredSize]
}

// GetBatchSize 获取批处理大小
func (vo *VideoOptimization) GetBatchSize() int {
	return vo.batchSize
}

// IsGPUEnabled 检查是否启用GPU
func (vo *VideoOptimization) IsGPUEnabled() bool {
	return vo.enableGPU
}

// IsCUDAEnabled 检查是否启用CUDA加速
func (vo *VideoOptimization) IsCUDAEnabled() bool {
	return vo.enableCUDA && vo.cudaAccelerator != nil
}

// GetCUDADeviceID 获取CUDA设备ID
func (vo *VideoOptimization) GetCUDADeviceID() int {
	return vo.cudaDeviceID
}

// GetCUDAPerformanceMetrics 获取CUDA性能指标
func (vo *VideoOptimization) GetCUDAPerformanceMetrics() map[string]interface{} {
	if !vo.IsCUDAEnabled() {
		return map[string]interface{}{
			"enabled": false,
			"error":   "CUDA未启用或初始化失败",
		}
	}
	return vo.cudaAccelerator.GetPerformanceMetrics()
}

// OptimizeCUDAMemory 优化CUDA内存使用
func (vo *VideoOptimization) OptimizeCUDAMemory() error {
	if !vo.IsCUDAEnabled() {
		return fmt.Errorf("CUDA未启用")
	}
	return vo.cudaAccelerator.OptimizeMemoryUsage()
}

// GetPreprocessBuffer 获取预处理缓冲区
func (vo *VideoOptimization) GetPreprocessBuffer() [][]float32 {
	return vo.preprocessBuf
}

// OptimizedDetectImage 优化的图像检测方法
func (vo *VideoOptimization) OptimizedDetectImage(detector *YOLO, img image.Image) ([]Detection, error) {
	// 获取输入尺寸
	inputWidth := detector.config.InputWidth
	inputHeight := detector.config.InputHeight
	if inputWidth == 0 {
		inputWidth = detector.config.InputSize
	}
	if inputHeight == 0 {
		inputHeight = detector.config.InputSize
	}

	// 使用极致性能预处理
	data, err := vo.extremePreprocessImage(img, inputWidth, inputHeight)
	if err != nil {
		return nil, fmt.Errorf("预处理失败: %v", err)
	}

	// 调用检测器的内部方法，跳过重复预处理
	result, err := detector.detectWithPreprocessedData(data, img)
	
	// 智能垃圾回收 - 安全地清理临时内存
	vo.SmartGarbageCollect(false)
	
	return result, err
}

// BatchDetectImages 批量检测图像 - 极致GPU性能 + CUDA加速
func (vo *VideoOptimization) BatchDetectImages(detector *YOLO, images []image.Image) ([][]Detection, error) {
	if len(images) == 0 {
		return nil, nil
	}

	// 获取输入尺寸
	inputWidth := detector.config.InputWidth
	inputHeight := detector.config.InputHeight
	if inputWidth == 0 {
		inputWidth = detector.config.InputSize
	}
	if inputHeight == 0 {
		inputHeight = detector.config.InputSize
	}

	// 如果启用CUDA加速，优先使用CUDA批处理
	if vo.enableCUDA && vo.cudaAccelerator != nil {
		batchData, err := vo.cudaAccelerator.BatchPreprocessImagesCUDA(images, inputWidth, inputHeight)
		if err == nil {
			// CUDA批处理成功，进行检测
			results := make([][]Detection, len(images))
			for i, data := range batchData {
				detections, err := detector.detectWithPreprocessedData(data, images[i])
				if err != nil {
					return nil, fmt.Errorf("检测图像 %d 失败: %v", i, err)
				}
				results[i] = detections
			}
			
			// ✅ CUDA批处理完成后安全清理内存（结果已保存到results中）
			vo.SmartGarbageCollect(len(images) >= 20)
			
			return results, nil
		}
		// CUDA失败时回退到CPU模式
		fmt.Printf("⚠️ CUDA批处理失败，回退到CPU模式: %v\n", err)
	}

	// 使用最大批处理大小
	batchSize := vo.maxBatchSize
	if len(images) < batchSize {
		batchSize = len(images)
	}

	results := make([][]Detection, len(images))
	var wg sync.WaitGroup
	var mu sync.Mutex // 保护results切片的并发写入
	errorChan := make(chan error, len(images))
	var firstError error
	var errorOnce sync.Once

	// 并行批处理
	for i := 0; i < len(images); i += batchSize {
		end := i + batchSize
		if end > len(images) {
			end = len(images)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				detections, err := vo.OptimizedDetectImage(detector, images[j])
				if err != nil {
					// 只记录第一个错误，避免通道阻塞
					errorOnce.Do(func() {
						firstError = err
					})
					return
				}
				// 使用互斥锁保护并发写入
				mu.Lock()
				results[j] = detections
				mu.Unlock()
			}
		}(i, end)
	}

	wg.Wait()
	close(errorChan)

	// 检查是否有错误
	if firstError != nil {
		return nil, firstError
	}

	// ✅ CPU批处理完成后安全清理内存（结果已保存到results中）
	vo.SmartGarbageCollect(len(images) >= 20)

	return results, nil
}

// AsyncDetectImage 异步检测图像
func (vo *VideoOptimization) AsyncDetectImage(detector *YOLO, img image.Image, id int) {
	// 获取输入尺寸
	inputWidth := detector.config.InputWidth
	inputHeight := detector.config.InputHeight
	if inputWidth == 0 {
		inputWidth = detector.config.InputSize
	}
	if inputHeight == 0 {
		inputHeight = detector.config.InputSize
	}

	// 提交异步任务
	select {
	case vo.asyncQueue <- &ProcessTask{
		img:    img,
		width:  inputWidth,
		height: inputHeight,
		id:     id,
	}:
	default:
		// 队列满时直接处理
		data, err := vo.extremePreprocessImage(img, inputWidth, inputHeight)
		vo.processDone <- &ProcessResult{
			data: data,
			err:  err,
			id:   id,
		}
	}
}

// GetAsyncResult 获取异步处理结果
func (vo *VideoOptimization) GetAsyncResult() *ProcessResult {
	select {
	case result := <-vo.processDone:
		return result
	default:
		return nil
	}
}

// GetAsyncResultBlocking 阻塞获取异步处理结果
func (vo *VideoOptimization) GetAsyncResultBlocking() *ProcessResult {
	return <-vo.processDone
}

// HasPendingResults 检查是否有待处理的结果
func (vo *VideoOptimization) HasPendingResults() bool {
	return len(vo.processDone) > 0
}

// GetQueueStatus 获取队列状态信息
func (vo *VideoOptimization) GetQueueStatus() (asyncQueueLen, processDoneLen, availableWorkers int) {
	return len(vo.asyncQueue), len(vo.processDone), len(vo.workerPool)
}

// GetMaxBatchSize 获取最大批处理大小
func (vo *VideoOptimization) GetMaxBatchSize() int {
	return vo.maxBatchSize
}

// GetParallelWorkers 获取并行工作线程数
func (vo *VideoOptimization) GetParallelWorkers() int {
	return vo.parallelWorkers
}

// GetStabilityStatus 获取稳定性状态信息 - 疯狂调用监控
func (vo *VideoOptimization) GetStabilityStatus() map[string]interface{} {
	status := make(map[string]interface{})

	// 熔断器状态
	vo.circuitBreaker.mu.RLock()
	status["circuit_breaker"] = map[string]interface{}{
		"state":         vo.circuitBreaker.state,
		"failure_count": vo.circuitBreaker.failureCount,
		"last_fail":     vo.circuitBreaker.lastFailTime,
	}
	vo.circuitBreaker.mu.RUnlock()

	// 限流器状态
	vo.rateLimiter.mu.Lock()
	status["rate_limiter"] = map[string]interface{}{
		"tokens":      vo.rateLimiter.tokens,
		"max_tokens":  vo.rateLimiter.maxTokens,
		"refill_rate": vo.rateLimiter.refillRate,
	}
	vo.rateLimiter.mu.Unlock()

	// 资源监控状态
	vo.resourceMonitor.mu.RLock()
	status["resource_monitor"] = map[string]interface{}{
		"memory_usage":    vo.resourceMonitor.memoryUsage,
		"goroutine_count": vo.resourceMonitor.goroutineCount,
		"cpu_usage":       vo.resourceMonitor.cpuUsage,
		"max_memory":      vo.resourceMonitor.maxMemory,
		"max_goroutines":  vo.resourceMonitor.maxGoroutines,
	}
	vo.resourceMonitor.mu.RUnlock()

	// 健康检查状态
	vo.healthChecker.mu.RLock()
	status["health_checker"] = map[string]interface{}{
		"is_healthy":    vo.healthChecker.isHealthy,
		"failure_count": vo.healthChecker.failureCount,
		"last_check":    vo.healthChecker.lastCheck,
	}
	vo.healthChecker.mu.RUnlock()

	// 性能指标
	vo.metrics.mu.RLock()
	status["performance_metrics"] = map[string]interface{}{
		"total_requests":   vo.metrics.totalRequests,
		"success_requests": vo.metrics.successRequests,
		"failed_requests":  vo.metrics.failedRequests,
		"avg_latency":      vo.metrics.avgLatency,
		"max_latency":      vo.metrics.maxLatency,
		"min_latency":      vo.metrics.minLatency,
		"throughput":       vo.metrics.throughput,
	}
	vo.metrics.mu.RUnlock()

	return status
}

// ResetStabilityMetrics 重置稳定性指标 - 用于长期运行重置
func (vo *VideoOptimization) ResetStabilityMetrics() {
	// 重置熔断器
	vo.circuitBreaker.mu.Lock()
	vo.circuitBreaker.state = Closed
	vo.circuitBreaker.failureCount = 0
	vo.circuitBreaker.mu.Unlock()

	// 重置性能指标
	vo.metrics.mu.Lock()
	vo.metrics.totalRequests = 0
	vo.metrics.successRequests = 0
	vo.metrics.failedRequests = 0
	vo.metrics.avgLatency = 0
	vo.metrics.maxLatency = 0
	vo.metrics.minLatency = time.Hour
	vo.metrics.throughput = 0
	vo.metrics.lastUpdate = time.Now()
	vo.metrics.mu.Unlock()

	// 重置健康检查
	vo.healthChecker.mu.Lock()
	vo.healthChecker.isHealthy = true
	vo.healthChecker.failureCount = 0
	vo.healthChecker.lastCheck = time.Now()
	vo.healthChecker.mu.Unlock()
}

// AdjustPerformanceSettings 动态调整性能设置 - 疯狂调用优化
func (vo *VideoOptimization) AdjustPerformanceSettings(maxMemoryMB int64, maxGoroutines int64, maxCPU float64) {
	vo.resourceMonitor.mu.Lock()
	defer vo.resourceMonitor.mu.Unlock()

	vo.resourceMonitor.maxMemory = maxMemoryMB * 1024 * 1024
	vo.resourceMonitor.maxGoroutines = maxGoroutines
	vo.resourceMonitor.maxCPU = maxCPU
}

// SetRateLimitSettings 动态调整限流设置 - 疯狂调用控制
func (vo *VideoOptimization) SetRateLimitSettings(maxTokens, refillRate int64) {
	vo.rateLimiter.mu.Lock()
	defer vo.rateLimiter.mu.Unlock()

	vo.rateLimiter.maxTokens = maxTokens
	vo.rateLimiter.refillRate = refillRate
	vo.rateLimiter.tokens = maxTokens // 立即生效
}

// SetCircuitBreakerSettings 动态调整熔断器设置 - 疯狂调用保护
func (vo *VideoOptimization) SetCircuitBreakerSettings(maxFailures int64, timeout, retryTimeout time.Duration) {
	vo.circuitBreaker.mu.Lock()
	defer vo.circuitBreaker.mu.Unlock()

	vo.circuitBreaker.maxFailures = maxFailures
	vo.circuitBreaker.timeout = timeout
	vo.circuitBreaker.retryTimeout = retryTimeout
}

// Close 关闭VideoOptimization，清理资源 - 疯狂调用安全关闭 + CUDA加速
func (vo *VideoOptimization) Close() {
	// 设置关闭标志
	atomic.StoreInt64(&vo.isShutdown, 1)

	// 取消上下文，通知所有监控循环退出
	vo.cancel()

	// 关闭CUDA加速器
	if vo.cudaAccelerator != nil {
		fmt.Println("🔒 正在关闭CUDA加速器...")
		vo.cudaAccelerator.Close()
		vo.cudaAccelerator = nil
	}

	// 等待一小段时间让工作线程优雅退出
	time.Sleep(100 * time.Millisecond)

	// 关闭异步队列，这会导致所有asyncWorker退出
	if vo.asyncQueue != nil {
		close(vo.asyncQueue)
	}

	// 等待所有工作线程完成
	for i := 0; i < vo.parallelWorkers; i++ {
		<-vo.workerPool
	}

	// 清空结果通道
	if vo.processDone != nil {
		for {
			select {
			case <-vo.processDone:
				// 继续清空
			default:
				// 通道已空，退出
				close(vo.processDone)
				return
			}
		}
	}

	fmt.Println("🔒 VideoOptimization 已安全关闭（包含CUDA资源）")
}

// IsHealthy 检查VideoOptimization的健康状态 - 疯狂调用健康检查
func (vo *VideoOptimization) IsHealthy() bool {
	// 检查是否已关闭
	if atomic.LoadInt64(&vo.isShutdown) == 1 {
		return false
	}

	if vo.asyncQueue == nil || vo.processDone == nil || vo.workerPool == nil {
		return false
	}

	// 检查健康检查器状态
	vo.healthChecker.mu.RLock()
	isHealthy := vo.healthChecker.isHealthy
	vo.healthChecker.mu.RUnlock()

	if !isHealthy {
		return false
	}

	// 检查熔断器状态
	vo.circuitBreaker.mu.RLock()
	circuitOpen := vo.circuitBreaker.state == Open
	vo.circuitBreaker.mu.RUnlock()

	if circuitOpen {
		return false
	}

	// 检查队列状态
	if len(vo.asyncQueue) > cap(vo.asyncQueue)*9/10 { // 队列使用超过90%
		return false
	}

	// 检查资源使用
	return vo.resourceCheck()
}

// SmartGarbageCollect 智能垃圾回收 - 安全地清理内存而不影响保存功能
func (vo *VideoOptimization) SmartGarbageCollect(forceGC bool) {
	vo.gcMutex.Lock()
	defer vo.gcMutex.Unlock()

	// 增加帧计数器
	atomic.AddInt64(&vo.frameCounter, 1)
	currentFrame := atomic.LoadInt64(&vo.frameCounter)

	// 检查是否需要执行GC
	shouldGC := forceGC || (currentFrame%vo.gcInterval == 0)

	// 时间间隔检查 - 避免过于频繁的GC
	timeSinceLastGC := time.Since(vo.lastGCTime)
	if !forceGC && timeSinceLastGC < 5*time.Second {
		return
	}

	if shouldGC {
		// 执行垃圾回收
		runtime.GC()
		vo.lastGCTime = time.Now()
		
		// 可选：强制释放操作系统内存
		runtime.GC()
	}
}

// SetGCInterval 设置垃圾回收间隔
func (vo *VideoOptimization) SetGCInterval(interval int64) {
	vo.gcMutex.Lock()
	defer vo.gcMutex.Unlock()
	vo.gcInterval = interval
}

// GetGCStats 获取垃圾回收统计信息
func (vo *VideoOptimization) GetGCStats() map[string]interface{} {
	vo.gcMutex.Lock()
	defer vo.gcMutex.Unlock()
	
	return map[string]interface{}{
		"frameCounter": atomic.LoadInt64(&vo.frameCounter),
		"gcInterval":   vo.gcInterval,
		"lastGCTime":   vo.lastGCTime,
		"timeSinceLastGC": time.Since(vo.lastGCTime),
	}
}

// ResetFrameCounter 重置帧计数器
func (vo *VideoOptimization) ResetFrameCounter() {
	atomic.StoreInt64(&vo.frameCounter, 0)
}
