package gui

import (
	"fmt"
	"image"
	"image/color"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/widget"
	"golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/Cubiaa/yolo-go/yolo"

	"github.com/disintegration/imaging"
)

// 输入源类型常量
const (
	InputTypeFile    = "file"    // 文件输入
	InputTypeVideo   = "video"   // 视频文件输入
	InputTypeCamera  = "camera"  // 摄像头输入
	InputTypeCam     = "cam"     // 摄像头输入（简写）
	InputTypeRTSP    = "rtsp"    // RTSP流输入
	InputTypeRTMP    = "rtmp"    // RTMP流输入
	InputTypeScreen  = "screen"  // 屏幕录制输入
	InputTypeDesktop = "desktop" // 桌面录制输入
)

// YOLOLiveWindow 实时视频播放窗口
type YOLOLiveWindow struct {
	app          fyne.App
	window       fyne.Window
	detector     *yolo.YOLO
	imageDisplay *canvas.Image
	statusLabel  *widget.Label
	fpsLabel     *widget.Label

	// 输入源信息
	inputSource *yolo.InputSource
	videoPath   string
	isPlaying   bool
	stopChan    chan bool

	// 检测配置
	drawBoxes     bool
	drawLabels    bool
	confThreshold float64
	iouThreshold  float64
	boxColor      string
	labelColor    string
	lineWidth     int
	fontSize      int
	showFPS       bool

	// 性能配置
	performanceMode string // "fast", "balanced", "accurate"
	frameSkip       int
	maxImageWidth   int
	maxImageHeight  int

	// 性能统计
	frameCount int
	startTime  time.Time
	fps        float64
}

// NewYOLOLiveWindow 创建实时视频播放窗口
func NewYOLOLiveWindow(detector *yolo.YOLO, inputType string, inputPath string, options *yolo.DetectionOptions) *YOLOLiveWindow {
	var inputSource *yolo.InputSource

	// 根据明确的输入源类型创建输入源
	switch strings.ToLower(inputType) {
	case InputTypeFile, InputTypeVideo:
		inputSource = yolo.NewFileInput(inputPath)
	case InputTypeCamera, InputTypeCam:
		inputSource = yolo.NewCameraInput(inputPath)
	case InputTypeRTSP:
		inputSource = yolo.NewRTSPInput(inputPath)
	case InputTypeRTMP:
		inputSource = yolo.NewRTMPInput(inputPath)
	case InputTypeScreen, InputTypeDesktop:
		inputSource = yolo.NewScreenInput()
	default:
		// 如果类型未知，默认为文件输入
		inputSource = yolo.NewFileInput(inputPath)
	}

	// 设置默认值，然后使用options中的值覆盖
	boxColor := "red"
	if options.BoxColor != "" {
		boxColor = options.BoxColor
	}
	
	labelColor := "white"
	if options.LabelColor != "" {
		labelColor = options.LabelColor
	}
	
	lineWidth := 2
	if options.LineWidth > 0 {
		lineWidth = options.LineWidth
	}
	
	fontSize := 12
	if options.FontSize > 0 {
		fontSize = options.FontSize
	}

	window := &YOLOLiveWindow{
		app:           app.New(),
		detector:      detector,
		inputSource:   inputSource,
		videoPath:     inputPath,
		drawBoxes:     options.DrawBoxes,
		drawLabels:    options.DrawLabels,
		confThreshold: float64(options.ConfThreshold),
		iouThreshold:  float64(options.IOUThreshold),
		boxColor:      boxColor,
		labelColor:    labelColor,
		lineWidth:     lineWidth,
		fontSize:      fontSize,
		showFPS:       options.ShowFPS,
		stopChan:      make(chan bool),

		// 性能配置 - 针对高性能CPU优化
		performanceMode: "fast",
		frameSkip:       1, // 减少跳帧以提高流畅度
		maxImageWidth:   1024,
		maxImageHeight:  768,
	}

	window.createWindow()

	// 自动开始播放
	go func() {
		// 延迟一点启动，确保窗口已经显示
		time.Sleep(500 * time.Millisecond)
		window.startPlayback()
	}()

	return window
}

// detectInputType 自动检测输入源类型
func detectInputType(inputPath string) string {
	// 网络流检测
	if strings.HasPrefix(inputPath, "rtsp://") {
		return InputTypeRTSP
	}
	if strings.HasPrefix(inputPath, "rtmp://") {
		return InputTypeRTMP
	}

	// 屏幕录制检测
	if inputPath == "screen" || inputPath == "desktop" {
		return InputTypeScreen
	}

	// 摄像头检测
	if isCameraInput(inputPath) {
		return InputTypeCamera
	}

	// 默认为文件输入
	return InputTypeFile
}

// isCameraInput 检测是否为摄像头输入
func isCameraInput(inputPath string) bool {
	// 摄像头关键字
	cameraKeywords := []string{
		"camera",
		"cam",
		"webcam",
	}

	// 摄像头设备路径模式
	cameraPatterns := []string{
		"video=",     // Windows: video=0, video=1
		"/dev/video", // Linux: /dev/video0, /dev/video1
		"dshow:",     // Windows DirectShow
		"vfwcap:",    // Windows Video for Windows
	}

	// 数字索引 (0, 1, 2, 3...)
	if len(inputPath) == 1 && inputPath >= "0" && inputPath <= "9" {
		return true
	}

	// 检查关键字
	for _, keyword := range cameraKeywords {
		if strings.EqualFold(inputPath, keyword) {
			return true
		}
	}

	// 检查设备路径模式
	for _, pattern := range cameraPatterns {
		if strings.HasPrefix(strings.ToLower(inputPath), pattern) {
			return true
		}
	}

	return false
}

// createWindow 创建窗口
func (live *YOLOLiveWindow) createWindow() {
	// 根据输入源类型设置窗口标题
	var windowTitle string
	switch live.inputSource.Type {
	case "file":
		windowTitle = fmt.Sprintf("YOLO 实时检测 - 文件: %s", live.videoPath)
	case "camera":
		windowTitle = fmt.Sprintf("YOLO 实时检测 - 摄像头: %s", live.inputSource.Path)
	case "rtsp":
		windowTitle = "YOLO 实时检测 - RTSP流"
	case "rtmp":
		windowTitle = "YOLO 实时检测 - RTMP流"
	case "screen":
		windowTitle = fmt.Sprintf("YOLO 实时检测 - 屏幕录制: %s", live.inputSource.Path)
	default:
		windowTitle = "YOLO 实时检测"
	}
	windowTitle += " (自动播放)"

	live.window = live.app.NewWindow(windowTitle)
	live.window.Resize(fyne.NewSize(1000, 700))

	// 创建图像显示区域
	live.imageDisplay = &canvas.Image{}
	live.imageDisplay.FillMode = canvas.ImageFillContain
	live.imageDisplay.SetMinSize(fyne.NewSize(800, 600))
	
	// 创建一个测试图像来验证显示功能
	testImg := image.NewRGBA(image.Rect(0, 0, 320, 240))
	for y := 0; y < 240; y++ {
		for x := 0; x < 320; x++ {
			// 创建一个简单的渐变图案
			r := uint8(x * 255 / 320)
			g := uint8(y * 255 / 240)
			b := uint8(128)
			testImg.Set(x, y, color.RGBA{r, g, b, 255})
		}
	}
	live.imageDisplay.Image = testImg
	fmt.Printf("设置了测试图像，尺寸: %dx%d\n", testImg.Bounds().Dx(), testImg.Bounds().Dy())

	// 创建状态标签
	live.statusLabel = widget.NewLabel("准备自动播放...")
	live.fpsLabel = widget.NewLabel("FPS: 0")

	// 创建性能模式选择
	performanceSelect := widget.NewSelect([]string{"fast", "balanced", "accurate"}, func(value string) {
		live.performanceMode = value
		live.statusLabel.SetText(fmt.Sprintf("性能模式: %s", value))
	})
	performanceSelect.SetSelected(live.performanceMode)

	// 创建控制按钮
	playBtn := widget.NewButton("播放", live.startPlayback)
	stopBtn := widget.NewButton("停止", live.stopPlayback)

	// 创建设备信息标签
	deviceInfo := widget.NewLabel(fmt.Sprintf("设备: %s", live.inputSource.Path))

	// 创建布局
	controls := container.NewHBox(playBtn, stopBtn, widget.NewLabel("性能模式:"), performanceSelect, live.statusLabel, live.fpsLabel)
	infoPanel := container.NewHBox(deviceInfo)
	content := container.NewVBox(live.imageDisplay, controls, infoPanel)

	live.window.SetContent(content)

	// 窗口关闭时停止播放
	live.window.SetOnClosed(func() {
		live.stopPlayback()
	})
}

// startPlayback 开始播放
func (live *YOLOLiveWindow) startPlayback() {
	if live.isPlaying {
		return
	}

	live.isPlaying = true
	live.startTime = time.Now()
	live.frameCount = 0

	fyne.Do(func() {
		live.statusLabel.SetText("正在播放...")
	})

	// 在后台运行视频处理
	go live.processVideo()
}

// stopPlayback 停止播放
func (live *YOLOLiveWindow) stopPlayback() {
	live.isPlaying = false
	live.stopChan <- true

	fyne.Do(func() {
		live.statusLabel.SetText("已停止")
	})
}

// processVideo 处理视频
func (live *YOLOLiveWindow) processVideo() {
	// 设置检测器的运行时配置，确保使用正确的置信度和IOU阈值
	detectionOptions := &yolo.DetectionOptions{
		ConfThreshold: float32(live.confThreshold),
		IOUThreshold:  float32(live.iouThreshold),
		DrawBoxes:     live.drawBoxes,
		DrawLabels:    live.drawLabels,
		ShowFPS:       live.showFPS,
		BoxColor:      live.boxColor,
		LabelColor:    live.labelColor,
		LineWidth:     live.lineWidth,
		FontSize:      live.fontSize,
	}
	
	// 将配置设置到检测器中
	live.detector.SetRuntimeConfig(detectionOptions)

	// 性能模式参数已在SetPerformanceMode中设置

	frameCount := 0
	// 创建检测结果通道，用于异步处理
	detectionChan := make(chan struct{
		img image.Image
		detections []yolo.Detection
		frameNum int
	}, 2) // 缓冲2帧，避免阻塞

	// 启动UI更新协程
	go func() {
		for result := range detectionChan {
			if !live.isPlaying {
				return
			}

			live.frameCount = result.frameNum
			fmt.Printf("GUI更新协程收到第 %d 帧，图像尺寸: %dx%d\n", result.frameNum, result.img.Bounds().Dx(), result.img.Bounds().Dy())

			// 计算FPS
			elapsed := time.Since(live.startTime).Seconds()
			if elapsed > 0 {
				live.fps = float64(live.frameCount) / elapsed
			}

			// 在主线程中更新UI
			fyne.Do(func() {
				fmt.Printf("开始更新GUI显示，帧号: %d\n", result.frameNum)
				
				// 更新FPS显示
				if live.showFPS {
					live.fpsLabel.SetText(fmt.Sprintf("FPS: %.1f", live.fps))
				}

				// 在图像上绘制检测结果
				processedImage := live.drawDetectionsOnImage(result.img, result.detections)
				fmt.Printf("图像处理完成，处理后图像尺寸: %dx%d\n", processedImage.Bounds().Dx(), processedImage.Bounds().Dy())

				// 更新显示
				live.imageDisplay.Image = processedImage
				live.imageDisplay.Refresh()
				fmt.Printf("GUI显示已更新，帧号: %d\n", result.frameNum)

				// 更新状态
				if live.inputSource.GetInputType() == "camera" {
					live.statusLabel.SetText(fmt.Sprintf("摄像头帧: %d, 检测: %d", live.frameCount, len(result.detections)))
				} else {
					live.statusLabel.SetText(fmt.Sprintf("帧: %d, 检测: %d", live.frameCount, len(result.detections)))
				}
			})
		}
	}()

	// 根据输入类型选择合适的处理器
	if live.inputSource.GetInputType() == "camera" {
		// 使用专门的摄像头处理器
		cameraProcessor := yolo.NewCameraVideoProcessor(live.detector, live.inputSource.Path)
		
		err := cameraProcessor.ProcessCameraWithCallback(func(result yolo.VideoDetectionResult) {
			if !live.isPlaying {
				return
			}

			frameCount++

			// 每5帧输出一次调试信息
			if frameCount%5 == 1 {
				fmt.Printf("GUI收到第 %d 帧，图像尺寸: %dx%d，检测数量: %d\n", frameCount, result.Image.Bounds().Dx(), result.Image.Bounds().Dy(), len(result.Detections))
			}

			// 跳帧处理以提高性能
			if frameCount%live.frameSkip != 0 {
				return
			}

			// 异步发送检测结果到UI更新协程
			select {
			case detectionChan <- struct{
				img image.Image
				detections []yolo.Detection
				frameNum int
			}{img: result.Image, detections: result.Detections, frameNum: frameCount}:
				fmt.Printf("成功发送第 %d 帧到GUI更新通道\n", frameCount)
			default:
				fmt.Printf("GUI更新通道满，跳过第 %d 帧\n", frameCount)
			}

			// 控制播放速度 - 摄像头使用较慢的速度
			time.Sleep(100 * time.Millisecond) // 降低到10 FPS，减少处理负担
		})
		
		if err != nil {
			fyne.Do(func() {
				live.statusLabel.SetText(fmt.Sprintf("摄像头处理失败: %v", err))
			})
		}
	} else if live.inputSource.GetInputType() == "rtsp" {
		// 使用直接FFmpeg方式处理RTSP流
		inputPath := live.inputSource.GetFFmpegInput()
		
		_, err := live.detector.DetectFromRTSP(inputPath, detectionOptions, func(result yolo.VideoDetectionResult) {
			if !live.isPlaying {
				return
			}

			frameCount++

			// 跳帧处理以提高性能
			if frameCount%live.frameSkip != 0 {
				return
			}

			// 异步发送检测结果到UI更新协程
			select {
			case detectionChan <- struct{
				img image.Image
				detections []yolo.Detection
				frameNum int
			}{img: result.Image, detections: result.Detections, frameNum: frameCount}:
				fmt.Printf("成功发送第 %d 帧到GUI更新通道\n", frameCount)
			default:
				fmt.Printf("GUI更新通道满，跳过第 %d 帧\n", frameCount)
			}

			// 控制播放速度
			time.Sleep(33 * time.Millisecond) // 约30 FPS
		})
		
		if err != nil {
			fyne.Do(func() {
				live.statusLabel.SetText(fmt.Sprintf("RTSP处理失败: %v", err))
			})
		}
	} else if live.inputSource.GetInputType() == "rtmp" {
		// 使用直接FFmpeg方式处理RTMP流
		inputPath := live.inputSource.GetFFmpegInput()
		
		_, err := live.detector.DetectFromRTMP(inputPath, detectionOptions, func(result yolo.VideoDetectionResult) {
			if !live.isPlaying {
				return
			}

			frameCount++

			// 跳帧处理以提高性能
			if frameCount%live.frameSkip != 0 {
				return
			}

			// 异步发送检测结果到UI更新协程
			select {
			case detectionChan <- struct{
				img image.Image
				detections []yolo.Detection
				frameNum int
			}{img: result.Image, detections: result.Detections, frameNum: frameCount}:
				fmt.Printf("成功发送第 %d 帧到GUI更新通道\n", frameCount)
			default:
				fmt.Printf("GUI更新通道满，跳过第 %d 帧\n", frameCount)
			}

			// 控制播放速度
			time.Sleep(33 * time.Millisecond) // 约30 FPS
		})
		
		if err != nil {
			fyne.Do(func() {
				live.statusLabel.SetText(fmt.Sprintf("RTMP处理失败: %v", err))
			})
		}
	} else {
		// 使用原有的视频处理器处理文件等
		processor := yolo.NewVidioVideoProcessorWithOptions(live.detector, detectionOptions)
		inputPath := live.inputSource.GetFFmpegInput()

		err := processor.ProcessVideoWithCallback(inputPath, func(result yolo.VideoDetectionResult) {
			if !live.isPlaying {
				return
			}

			frameCount++

			// 跳帧处理以提高性能
			if frameCount%live.frameSkip != 0 {
				return
			}

			live.frameCount++

			// 计算FPS
			elapsed := time.Since(live.startTime).Seconds()
			if elapsed > 0 {
				live.fps = float64(live.frameCount) / elapsed
			}

			// 异步发送检测结果到UI更新协程
			if result.Image != nil {
				select {
				case detectionChan <- struct{
					img image.Image
					detections []yolo.Detection
					frameNum int
				}{img: result.Image, detections: result.Detections, frameNum: frameCount}:
				default:
					// 如果通道满了，跳过这一帧，避免阻塞
				}
			}

			// 控制播放速度 - 针对高性能CPU优化
			time.Sleep(8 * time.Millisecond) // 约120 FPS，更流畅
		})

		if err != nil {
			fyne.Do(func() {
				live.statusLabel.SetText(fmt.Sprintf("处理失败: %v", err))
			})
		}
	}
}

// drawDetectionsOnImage 在图像上绘制检测结果
func (live *YOLOLiveWindow) drawDetectionsOnImage(img image.Image, detections []yolo.Detection) image.Image {
	// 极限性能模式：移除调试输出以提升GUI响应速度
	
	// 获取原始图像尺寸
	originalBounds := img.Bounds()
	originalWidth := float32(originalBounds.Dx())
	originalHeight := float32(originalBounds.Dy())
	
	// 性能优化：缩放图像以提高处理速度（针对高性能CPU优化）
	var scale float32 = 1.0
	var targetWidth, targetHeight int
	
	// 根据性能模式动态调整显示尺寸
	switch live.performanceMode {
	case "fast":
		targetWidth, targetHeight = 640, 480
	case "balanced":
		targetWidth, targetHeight = 1024, 768
	case "accurate":
		targetWidth, targetHeight = live.maxImageWidth, live.maxImageHeight
	}
	
	if originalBounds.Dx() > targetWidth || originalBounds.Dy() > targetHeight {
		// 计算缩放比例
		scaleX := float32(targetWidth) / originalWidth
		scaleY := float32(targetHeight) / originalHeight
		scale = scaleX
		if scaleY < scaleX {
			scale = scaleY
		}

		// 缩放图像
		newWidth := int(originalWidth * scale)
		newHeight := int(originalHeight * scale)
		img = imaging.Resize(img, newWidth, newHeight, imaging.Lanczos)
	}

	// 转换为RGBA
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)
	draw.Draw(result, bounds, img, bounds.Min, draw.Src)

	// 绘制检测框和标签（需要根据缩放比例调整坐标）
	for _, detection := range detections {
		// 调整检测框坐标以匹配缩放后的图像
		scaledBox := [4]float32{
			detection.Box[0] * scale, // x1
			detection.Box[1] * scale, // y1
			detection.Box[2] * scale, // x2
			detection.Box[3] * scale, // y2
		}
		
		if live.drawBoxes {
			live.drawBox(result, scaledBox, live.getColor(live.boxColor))
		}
		if live.drawLabels {
			live.drawLabel(result, detection.Class, detection.Score, scaledBox)
		}
	}

	return result
}

// drawBox 绘制检测框
func (live *YOLOLiveWindow) drawBox(img *image.RGBA, box [4]float32, color color.Color) {
	// 简化的绘制逻辑
	x1, y1, x2, y2 := int(box[0]), int(box[1]), int(box[2]), int(box[3])

	// 确保坐标在图像范围内
	bounds := img.Bounds()
	if x1 < bounds.Min.X {
		x1 = bounds.Min.X
	}
	if y1 < bounds.Min.Y {
		y1 = bounds.Min.Y
	}
	if x2 > bounds.Max.X {
		x2 = bounds.Max.X
	}
	if y2 > bounds.Max.Y {
		y2 = bounds.Max.Y
	}

	// 检查框的有效性
	if x2 <= x1 || y2 <= y1 {
		return // 无效的框，不绘制
	}

	// 绘制矩形边框 - 使用更粗的线条
	lineWidth := live.lineWidth
	if lineWidth < 1 {
		lineWidth = 2
	}

	// 绘制上边框
	for x := x1; x <= x2; x++ {
		for w := 0; w < lineWidth && y1+w < bounds.Max.Y; w++ {
			img.Set(x, y1+w, color)
		}
	}

	// 绘制下边框
	for x := x1; x <= x2; x++ {
		for w := 0; w < lineWidth && y2-w >= bounds.Min.Y; w++ {
			img.Set(x, y2-w, color)
		}
	}

	// 绘制左边框
	for y := y1; y <= y2; y++ {
		for w := 0; w < lineWidth && x1+w < bounds.Max.X; w++ {
			img.Set(x1+w, y, color)
		}
	}

	// 绘制右边框
	for y := y1; y <= y2; y++ {
		for w := 0; w < lineWidth && x2-w >= bounds.Min.X; w++ {
			img.Set(x2-w, y, color)
		}
	}
}

// drawLabel 绘制标签
func (live *YOLOLiveWindow) drawLabel(img *image.RGBA, className string, score float32, box [4]float32) {
	label := fmt.Sprintf("%s %.2f", className, score)
	x, y := int(box[0]), int(box[1])-20

	// 确保坐标在图像范围内
	bounds := img.Bounds()
	if x < bounds.Min.X {
		x = bounds.Min.X
	}
	if y < bounds.Min.Y {
		y = bounds.Min.Y
	}

	// 不绘制文本背景，直接绘制文本

	// 绘制文本
	point := fixed.Point26_6{X: fixed.Int26_6(x * 64), Y: fixed.Int26_6(y * 64)}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(live.getColor(live.labelColor)),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

// getColor 获取颜色
func (live *YOLOLiveWindow) getColor(colorName string) color.Color {
	switch colorName {
	case "red":
		return color.RGBA{255, 0, 0, 255}
	case "green":
		return color.RGBA{0, 255, 0, 255}
	case "blue":
		return color.RGBA{0, 0, 255, 255}
	case "yellow":
		return color.RGBA{255, 255, 0, 255}
	case "white":
		return color.RGBA{255, 255, 255, 255}
	case "black":
		return color.RGBA{0, 0, 0, 255}
	default:
		return color.RGBA{255, 0, 0, 255}
	}
}

// Run 运行窗口
// SetPerformanceMode 设置性能模式
func (live *YOLOLiveWindow) SetPerformanceMode(mode string) {
	live.performanceMode = mode
	
	// 根据性能模式更新参数
	switch mode {
	case "fast":
		live.frameSkip = 2  // 跳帧处理，每2帧处理一次
		live.maxImageWidth = 320   // 更小的图像尺寸
		live.maxImageHeight = 320
		// 调整检测阈值以提升速度
		live.confThreshold = 0.5   // 更高的置信度阈值
		live.iouThreshold = 0.6    // 更高的IOU阈值
	case "balanced":
		live.frameSkip = 1
		live.maxImageWidth = 640
		live.maxImageHeight = 640
		live.confThreshold = 0.25
		live.iouThreshold = 0.45
	case "accurate":
		live.frameSkip = 1
		live.maxImageWidth = 832
		live.maxImageHeight = 832
		live.confThreshold = 0.15  // 更低的置信度阈值，检测更多对象
		live.iouThreshold = 0.4
	default:
		live.performanceMode = "balanced"
		live.frameSkip = 1
		live.maxImageWidth = 640
		live.maxImageHeight = 640
		live.confThreshold = 0.25
		live.iouThreshold = 0.45
	}
}

// GetPerformanceMode 获取当前性能模式
func (live *YOLOLiveWindow) GetPerformanceMode() string {
	return live.performanceMode
}

func (live *YOLOLiveWindow) Run() {
	live.window.ShowAndRun()
}

// LaunchFyneLiveWindow 启动Fyne实时窗口
func LaunchFyneLiveWindow(detector *yolo.YOLO, videoPath string, options *yolo.DetectionOptions) error {
	fmt.Printf("🎬 启动Fyne GUI窗口，视频: %s\n", videoPath)

	// 自动检测输入源类型
	inputType := detectInputType(videoPath)

	// 创建并运行GUI窗口
	liveWindow := NewYOLOLiveWindow(detector, inputType, videoPath, options)
	liveWindow.Run()

	return nil
}
