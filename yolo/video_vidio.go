package yolo

import (
	"fmt"
	"image"
	"time"

	vidio "github.com/AlexEidt/Vidio"
)

// VidioVideoProcessor 使用Vidio库的视频处理器
type VidioVideoProcessor struct {
	detector *YOLO
}

// NewVidioVideoProcessor 创建Vidio视频处理器
func NewVidioVideoProcessor(detector *YOLO) *VidioVideoProcessor {
	return &VidioVideoProcessor{
		detector: detector,
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
		detections, err := vp.detector.detectImage(frameImg)
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

// ProcessVideoWithCallback 处理视频并对每帧调用回调函数
func (vp *VidioVideoProcessor) ProcessVideoWithCallback(inputPath string, callback func(VideoDetectionResult)) error {
	// 打开视频文件
	video, err := vidio.NewVideo(inputPath)
	if err != nil {
		return fmt.Errorf("无法打开视频文件: %v", err)
	}
	defer video.Close()

	fmt.Printf("📹 视频信息: %dx%d, %.2f FPS, %d 帧\n",
		video.Width(), video.Height(), video.FPS(), video.Frames())

	frameCount := 0

	// 逐帧读取视频
	for video.Read() {
		frameCount++

		// 将帧缓冲区转换为Go图像
		frameImg := convertFrameBufferToImage(video.FrameBuffer(), video.Width(), video.Height())

		// YOLO检测
		detections, err := vp.detector.detectImage(frameImg)
		if err != nil {
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

		// 进度提示
		if frameCount%30 == 0 {
			fmt.Printf("📊 已处理 %d/%d 帧...\n", frameCount, video.Frames())
		}
	}

	fmt.Printf("✅ 视频处理完成！共处理 %d 帧\n", frameCount)
	return nil
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
		Quality: 0.8, // 高质量
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

// convertImageToFrameBuffer 将Go图像转换为帧缓冲区
func convertImageToFrameBuffer(img image.Image) []byte {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 创建RGBA图像
	rgba := image.NewRGBA(bounds)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}

	return rgba.Pix
}
