package yolo

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// SimpleVideoProcessor 简单视频处理器（使用标准库）
type SimpleVideoProcessor struct {
	detector *YOLO
}

// NewSimpleVideoProcessor 创建简单视频处理器
func NewSimpleVideoProcessor(detector *YOLO) *SimpleVideoProcessor {
	return &SimpleVideoProcessor{
		detector: detector,
	}
}

// ProcessImageSequence 处理图像序列
func (svp *SimpleVideoProcessor) ProcessImageSequence(inputDir, outputDir string) error {
	// 创建输出目录
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("创建输出目录失败: %v", err)
	}

	// 读取输入目录中的所有图像文件
	files, err := os.ReadDir(inputDir)
	if err != nil {
		return fmt.Errorf("读取输入目录失败: %v", err)
	}

	// 支持的图像格式
	supportedFormats := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
	}

	for i, file := range files {
		if file.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(file.Name()))
		if !supportedFormats[ext] {
			continue
		}

		inputPath := filepath.Join(inputDir, file.Name())
		outputPath := filepath.Join(outputDir, fmt.Sprintf("frame_%04d.jpg", i))

		// 检测图像
		detections, err := svp.detector.Detect(inputPath)
		if err != nil {
			fmt.Printf("处理图像 %s 失败: %v\n", inputPath, err)
			continue
		}

		// 读取原始图像
		img, err := loadImage(inputPath)
		if err != nil {
			fmt.Printf("读取图像 %s 失败: %v\n", inputPath, err)
			continue
		}

		// 绘制检测结果
		resultImg := svp.drawDetectionsOnImage(img, detections.Detections)

		// 保存结果
		if err := SaveImage(resultImg, outputPath); err != nil {
			fmt.Printf("保存图像 %s 失败: %v\n", outputPath, err)
			continue
		}

		fmt.Printf("处理完成: %s -> %s (检测到 %d 个对象)\n", file.Name(), outputPath, len(detections.Detections))
	}

	return nil
}

// ProcessImageWithCallback 处理图像并调用回调函数
func (svp *SimpleVideoProcessor) ProcessImageWithCallback(inputDir string, callback func(VideoDetectionResult)) error {
	files, err := os.ReadDir(inputDir)
	if err != nil {
		return fmt.Errorf("读取输入目录失败: %v", err)
	}

	supportedFormats := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
	}

	for i, file := range files {
		if file.IsDir() {
			continue
		}

		ext := strings.ToLower(filepath.Ext(file.Name()))
		if !supportedFormats[ext] {
			continue
		}

		inputPath := filepath.Join(inputDir, file.Name())

		// 检测图像
		detections, err := svp.detector.Detect(inputPath)
		if err != nil {
			fmt.Printf("处理图像 %s 失败: %v\n", inputPath, err)
			continue
		}

		// 调用回调函数
		result := VideoDetectionResult{
			FrameNumber: i,
			Timestamp:   time.Duration(i) * time.Second,
			Detections:  detections.Detections,
		}
		callback(result)
	}

	return nil
}

// drawDetectionsOnImage 在图像上绘制检测结果
func (svp *SimpleVideoProcessor) drawDetectionsOnImage(img image.Image, detections []Detection) image.Image {
	// 创建可绘制的图像副本
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	draw.Draw(result, bounds, img, bounds.Min, draw.Src)

	// 获取颜色配置
	boxColor := color.RGBA{255, 0, 0, 255} // 默认红色
	if svp.detector.runtimeConfig != nil && svp.detector.runtimeConfig.BoxColor != "" {
		if parsedColor := svp.detector.parseColor(svp.detector.runtimeConfig.BoxColor); parsedColor != nil {
			boxColor = *parsedColor
		}
	}

	// 绘制每个检测结果
	for _, detection := range detections {
		// 检查是否应该画框和标签
		drawBoxes := true
		drawLabels := true
		if svp.detector.runtimeConfig != nil {
			drawBoxes = svp.detector.runtimeConfig.DrawBoxes
			drawLabels = svp.detector.runtimeConfig.DrawLabels
		}

		if drawBoxes {
			svp.drawBBoxOnImage(result, detection.Box, boxColor)
		}

		if drawLabels {
			// 绘制标签文本
			label := fmt.Sprintf("%s %.2f", detection.Class, detection.Score)
			svp.drawLabelOnImage(result, label, detection.Box)
		}
	}

	return result
}

// drawBBoxOnImage 在图像上绘制边界框
func (svp *SimpleVideoProcessor) drawBBoxOnImage(img draw.Image, bbox [4]float32, lineColor color.Color) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// 计算边界框坐标
	x1 := int(bbox[0] * float32(width))
	y1 := int(bbox[1] * float32(height))
	x2 := int(bbox[2] * float32(width))
	y2 := int(bbox[3] * float32(height))

	// 确保坐标在图像范围内
	x1 = int(max(0, min(float32(x1), float32(width-1))))
	y1 = int(max(0, min(float32(y1), float32(height-1))))
	x2 = int(max(0, min(float32(x2), float32(width-1))))
	y2 = int(max(0, min(float32(y2), float32(height-1))))

	// 获取线条宽度
	lineWidth := 2
	if svp.detector.runtimeConfig != nil && svp.detector.runtimeConfig.LineWidth > 0 {
		lineWidth = svp.detector.runtimeConfig.LineWidth
	}

	// 绘制边界框
	for i := 0; i < lineWidth; i++ {
		// 上边
		for x := x1; x <= x2; x++ {
			if y1+i < height {
				img.Set(x, y1+i, lineColor)
			}
		}
		// 下边
		for x := x1; x <= x2; x++ {
			if y2-i >= 0 {
				img.Set(x, y2-i, lineColor)
			}
		}
		// 左边
		for y := y1; y <= y2; y++ {
			if x1+i < width {
				img.Set(x1+i, y, lineColor)
			}
		}
		// 右边
		for y := y1; y <= y2; y++ {
			if x2-i >= 0 {
				img.Set(x2-i, y, lineColor)
			}
		}
	}
}

// drawLabelOnImage 在图像上绘制标签
func (svp *SimpleVideoProcessor) drawLabelOnImage(img *image.RGBA, label string, bbox [4]float32) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// 计算标签位置
	x := int(bbox[0] * float32(width))
	y := int(bbox[1] * float32(height)) - 20

	// 确保标签在图像范围内
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 20
	}

	// 获取字体大小
	fontSize := 12
	if svp.detector.runtimeConfig != nil && svp.detector.runtimeConfig.FontSize > 0 {
		fontSize = svp.detector.runtimeConfig.FontSize
	}

	// 不绘制背景矩形，直接绘制文本

	// 获取标签颜色
	labelColor := color.RGBA{255, 255, 255, 255} // 默认白色
	if svp.detector.runtimeConfig != nil && svp.detector.runtimeConfig.LabelColor != "" {
		if parsedColor := svp.detector.parseColor(svp.detector.runtimeConfig.LabelColor); parsedColor != nil {
			labelColor = *parsedColor
		}
	}

	// 简单的文本绘制（使用像素点）
	// 这里使用简化的文本渲染，实际项目中可以使用更复杂的字体库
	for i, char := range label {
		charX := x + i*(fontSize/2)
		if charX >= 0 && charX < width-fontSize/2 && y >= 0 && y < height-fontSize {
			// 简单的字符渲染（仅作示例）
			for dy := 0; dy < fontSize; dy++ {
				for dx := 0; dx < fontSize/2; dx++ {
					px := charX + dx
					py := y + dy
					if px >= 0 && py >= 0 && px < width && py < height {
						// 简单的字符模式（可以根据需要改进）
						if (dy == 0 || dy == fontSize-1 || dx == 0 || dx == fontSize/2-1) && char != ' ' {
							img.Set(px, py, labelColor)
						}
					}
				}
			}
		}
	}
}

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// SaveImage 保存图像
func SaveImage(img image.Image, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".jpg", ".jpeg":
		return jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
	case ".png":
		return png.Encode(file, img)
	default:
		return jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
	}
}
