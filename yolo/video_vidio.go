package yolo

import (
	"fmt"
	"image"
	"image/draw"
	"time"

	vidio "github.com/AlexEidt/Vidio"
)

// VidioVideoProcessor 使用Vidio库的视频处理器
type VidioVideoProcessor struct {
	detector     *YOLO
	optimization *VideoOptimization
}

// NewVidioVideoProcessor 创建Vidio视频处理器
func NewVidioVideoProcessor(detector *YOLO) *VidioVideoProcessor {
	return &VidioVideoProcessor{
		detector:     detector,
		optimization: NewVideoOptimization(detector.config.UseGPU),
	}
}

// NewVidioVideoProcessorWithOptions 创建带配置选项的Vidio视频处理器
func NewVidioVideoProcessorWithOptions(detector *YOLO, options *DetectionOptions) *VidioVideoProcessor {
	return &VidioVideoProcessor{
		detector:     detector,
		optimization: NewVideoOptimization(detector.config.UseGPU),
	}
}

// ProcessVideo 处理视频文件并返回所有检测结果
func (vp *VidioVideoProcessor) ProcessVideo(inputPath string) ([]VideoDetectionResult, error) {
	// 打开视频文件
	video, err := vidio.NewVideo(inputPath)
	if err != nil {
		return nil, fmt.Errorf("无法打开视频文件: %v", err)
	}
	defer video.Close()

	fmt.Printf("📹 视频信息: %dx%d, %.2f FPS, %d 帧, %.2f 秒\n",
		video.Width(), video.Height(), video.FPS(), video.Frames(), video.Duration())

	var results []VideoDetectionResult
	frameCount := 0

	// 逐帧读取视频
	for video.Read() {
		frameCount++

		// 将帧缓冲区转换为Go图像
		frameImg := convertFrameBufferToImage(video.FrameBuffer(), video.Width(), video.Height())

		// YOLO检测
		var detections []Detection
		var err error
		

		
		detections, err = vp.detector.detectImage(frameImg)
		if err != nil {
			fmt.Printf("⚠️  帧 %d 检测失败: %v\n", frameCount, err)
			detections = []Detection{}
		}

		// 创建检测结果
		timestamp := time.Duration(float64(frameCount)/video.FPS()*1000) * time.Millisecond
		result := VideoDetectionResult{
			FrameNumber: frameCount,
			Timestamp:   timestamp,
			Detections:  detections,
			Image:       frameImg,
		}
		results = append(results, result)

		// 进度提示
		if frameCount%30 == 0 || frameCount == video.Frames() {
			fmt.Printf("📊 已处理 %d/%d 帧...\n", frameCount, video.Frames())
		}
	}

	fmt.Printf("✅ 视频处理完成！共处理 %d 帧\n", frameCount)
	return results, nil
}

// GetOptimization 获取视频优化实例
func (vp *VidioVideoProcessor) GetOptimization() *VideoOptimization {
	return vp.optimization
}

// ProcessVideoWithCallback 处理视频并对每帧调用回调函数（优化版本）
func (vp *VidioVideoProcessor) ProcessVideoWithCallback(inputPath string, callback func(VideoDetectionResult)) error {
	// 打开视频文件
	video, err := vidio.NewVideo(inputPath)
	if err != nil {
		return fmt.Errorf("无法打开视频文件: %v", err)
	}
	defer video.Close()

	fmt.Printf("📹 视频信息: %dx%d, %.2f FPS, %d 帧\n",
		video.Width(), video.Height(), video.FPS(), video.Frames())
	fmt.Printf("🚀 性能优化: 批处理大小=%d, GPU加速=%v\n", vp.optimization.GetBatchSize(), vp.optimization.IsGPUEnabled())

	frameCount := 0
	startTime := time.Now()



	// 逐帧读取视频（优化版本）
	for video.Read() {
		frameCount++

		// 将帧缓冲区转换为Go图像
		frameImg := convertFrameBufferToImage(video.FrameBuffer(), video.Width(), video.Height())

		// 使用优化的检测方法
		detections, err := vp.optimizedDetectImage(frameImg)
		if err != nil {
			// 减少错误输出频率
			if frameCount%100 == 0 {
				fmt.Printf("❌ 检测错误 (帧 %d): %v\n", frameCount, err)
			}
			detections = []Detection{}
		}

		// 创建检测结果并调用回调
		timestamp := time.Duration(float64(frameCount)/video.FPS()*1000) * time.Millisecond
		result := VideoDetectionResult{
			FrameNumber: frameCount,
			Timestamp:   timestamp,
			Detections:  detections,
			Image:       frameImg,
		}
		callback(result)

		// 性能监控和进度提示
		if frameCount%100 == 0 {
			elapsed := time.Since(startTime)
			fps := float64(frameCount) / elapsed.Seconds()
			fmt.Printf("📊 已处理 %d/%d 帧, 当前FPS: %.1f\n", frameCount, video.Frames(), fps)
		}
	}

	elapsed := time.Since(startTime)
	avgFPS := float64(frameCount) / elapsed.Seconds()
	fmt.Printf("✅ 视频处理完成！共处理 %d 帧, 平均FPS: %.1f, 总耗时: %v\n", frameCount, avgFPS, elapsed)
	return nil
}

// optimizedDetectImage 优化的图像检测方法
func (vp *VidioVideoProcessor) optimizedDetectImage(img image.Image) ([]Detection, error) {
	// 使用优化模块进行检测
	return vp.optimization.OptimizedDetectImage(vp.detector, img)
}

// SaveVideoWithDetections 保存带检测框的视频
func (vp *VidioVideoProcessor) SaveVideoWithDetections(inputPath, outputPath string) error {
	// 打开输入视频
	video, err := vidio.NewVideo(inputPath)
	if err != nil {
		return fmt.Errorf("无法打开视频文件: %v", err)
	}
	defer video.Close()

	// 创建输出视频写入器
	options := &vidio.Options{
		FPS:     video.FPS(),
		Quality: 1.0, // 无损质量，保持原画质
	}

	writer, err := vidio.NewVideoWriter(outputPath, video.Width(), video.Height(), options)
	if err != nil {
		return fmt.Errorf("无法创建输出视频: %v", err)
	}
	defer writer.Close()

	fmt.Printf("📹 开始处理视频: %s -> %s\n", inputPath, outputPath)
	frameCount := 0

	// 逐帧处理
	for video.Read() {
		frameCount++

		// 将帧缓冲区转换为Go图像
		frameImg := convertFrameBufferToImage(video.FrameBuffer(), video.Width(), video.Height())

		// YOLO检测
		detections, err := vp.detector.detectImage(frameImg)
		if err != nil {
			detections = []Detection{}
		}

		// 绘制检测结果
		var resultImg image.Image = frameImg
		if len(detections) > 0 {
			resultImg = vp.detector.drawDetectionsOnImage(frameImg, detections)
		}

		// 将图像转换回帧缓冲区并写入
		frameBuffer := convertImageToFrameBuffer(resultImg)
		err = writer.Write(frameBuffer)
		if err != nil {
			return fmt.Errorf("写入帧失败: %v", err)
		}

		// 进度提示
		if frameCount%30 == 0 {
			fmt.Printf("📊 已处理 %d/%d 帧...\n", frameCount, video.Frames())
		}
	}

	fmt.Printf("✅ 视频保存完成！共处理 %d 帧，保存为 %s\n", frameCount, outputPath)
	return nil
}

// convertFrameBufferToImage 将Vidio的帧缓冲区转换为Go图像
func convertFrameBufferToImage(frameBuffer []byte, width, height int) image.Image {
	// Vidio返回RGBA格式的字节数组
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	copy(img.Pix, frameBuffer)
	return img
}

// optimizedPreprocessImage 优化的图像预处理方法
func (vp *VidioVideoProcessor) optimizedPreprocessImage(img image.Image) ([]float32, error) {
	// 获取输入尺寸
	var inputWidth, inputHeight int
	if vp.detector.config.InputWidth > 0 && vp.detector.config.InputHeight > 0 {
		inputWidth = vp.detector.config.InputWidth
		inputHeight = vp.detector.config.InputHeight
	} else {
		inputWidth = vp.detector.config.InputSize
		inputHeight = vp.detector.config.InputSize
	}

	// 使用优化模块进行预处理
	return vp.optimization.OptimizedPreprocessImage(img, inputWidth, inputHeight)
}



// convertImageToFrameBuffer 将Go图像转换为帧缓冲区
func convertImageToFrameBuffer(img image.Image) []byte {
	bounds := img.Bounds()
	
	// 如果输入已经是RGBA格式，直接返回像素数据
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba.Pix
	}
	
	// 否则创建新的RGBA图像并高效复制
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	
	return rgba.Pix
}
