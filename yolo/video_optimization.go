package yolo

import (
	"fmt"
	"image"
	"runtime"
	"sync"

	"github.com/disintegration/imaging"
)

// VideoOptimization GPU优化相关的结构体和方法
type VideoOptimization struct {
	batchSize       int
	preprocessBuf   [][]float32
	imagePool       *sync.Pool
	enableGPU       bool
	// 极致性能优化字段
	maxBatchSize    int
	workerPool      chan struct{}
	preprocessPool  *sync.Pool
	resultPool      *sync.Pool
	parallelWorkers int
	memoryBuffer    [][]float32
	asyncQueue      chan *ProcessTask
	processDone     chan *ProcessResult
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

// NewVideoOptimization 创建视频优化实例 - 极致性能版本
func NewVideoOptimization(enableGPU bool) *VideoOptimization {
	// 极致性能配置 - 不计成本
	cpuCores := runtime.NumCPU()
	
	// 疯狂压榨GPU - 大幅增加批处理大小
	batchSize := cpuCores * 4 // 基础批处理
	maxBatchSize := cpuCores * 16 // 最大批处理，疯狂模式
	parallelWorkers := cpuCores * 8 // 并行工作线程数
	
	if enableGPU {
		// GPU模式下进一步增加批处理
		batchSize = cpuCores * 8
		maxBatchSize = cpuCores * 32 // GPU疯狂模式
		parallelWorkers = cpuCores * 16
	}

	// 预分配大量内存缓冲区
	preprocessBuf := make([][]float32, batchSize)
	memoryBuffer := make([][]float32, maxBatchSize)
	for i := range memoryBuffer {
		memoryBuffer[i] = make([]float32, 3*1024*1024) // 预分配1024x1024缓冲区
	}

	// 创建多个对象池用于极致性能
	imagePool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*1024*1024) // 更大的缓冲区
		},
	}
	
	preprocessPool := &sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*1024*1024)
		},
	}
	
	resultPool := &sync.Pool{
		New: func() interface{} {
			return make([]Detection, 0, 1000) // 预分配检测结果
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
	}
	
	// 启动异步处理工作线程
	vo.startAsyncWorkers()
	
	return vo
}

// startAsyncWorkers 启动异步处理工作线程
func (vo *VideoOptimization) startAsyncWorkers() {
	for i := 0; i < vo.parallelWorkers; i++ {
		go vo.asyncWorker()
	}
}

// asyncWorker 异步工作线程
func (vo *VideoOptimization) asyncWorker() {
	for task := range vo.asyncQueue {
		<-vo.workerPool // 获取工作许可
		
		// 执行预处理
		data, err := vo.extremePreprocessImage(task.img, task.width, task.height)
		
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
	}
}

// OptimizedPreprocessImage 优化的图像预处理方法 - 极致性能版本
func (vo *VideoOptimization) OptimizedPreprocessImage(img image.Image, inputWidth, inputHeight int) ([]float32, error) {
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

// fastResize 快速图像缩放
func (vo *VideoOptimization) fastResize(img image.Image, width, height int) image.Image {
	// 使用最快的缩放算法
	return imaging.Resize(img, width, height, imaging.NearestNeighbor)
}

// extremeFastResize 极致性能图像缩放
func (vo *VideoOptimization) extremeFastResize(img image.Image, width, height int) image.Image {
	// 检查是否需要缩放
	bounds := img.Bounds()
	if bounds.Dx() == width && bounds.Dy() == height {
		return img // 无需缩放，直接返回
	}
	
	// 使用最快的缩放算法 - NearestNeighbor
	return imaging.Resize(img, width, height, imaging.NearestNeighbor)
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
			buf[y*width+x] = float32(r>>8) / 255.0                    // R 通道
			buf[height*width+y*width+x] = float32(g>>8) / 255.0       // G 通道
			buf[2*height*width+y*width+x] = float32(b>>8) / 255.0     // B 通道
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
					buf[y*width+x] = float32(r>>8) / 255.0                    // R 通道
					buf[height*width+y*width+x] = float32(g>>8) / 255.0       // G 通道
					buf[2*height*width+y*width+x] = float32(b>>8) / 255.0     // B 通道
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
			buf[y*width+x] = float32(pix[i]) / 255.0                    // R通道
			buf[height*width+y*width+x] = float32(pix[i+1]) / 255.0     // G通道
			buf[2*height*width+y*width+x] = float32(pix[i+2]) / 255.0   // B通道
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
	return detector.detectWithPreprocessedData(data, img)
}

// BatchDetectImages 批量检测图像 - 极致GPU性能
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

// Close 关闭VideoOptimization，清理资源
func (vo *VideoOptimization) Close() {
	// 关闭异步队列，这会导致所有asyncWorker退出
	if vo.asyncQueue != nil {
		close(vo.asyncQueue)
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
}

// IsHealthy 检查VideoOptimization的健康状态
func (vo *VideoOptimization) IsHealthy() bool {
	if vo.asyncQueue == nil || vo.processDone == nil || vo.workerPool == nil {
		return false
	}
	
	// 检查是否有可用的工作线程
	return len(vo.workerPool) > 0
}