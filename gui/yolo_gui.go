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
func NewYOLOLiveWindow(detector *yolo.YOLO, inputPath string, options *yolo.DetectionOptions) *YOLOLiveWindow {
	// 根据输入路径创建输入源
	var inputSource *yolo.InputSource
	if strings.HasPrefix(inputPath, "rtsp://") {
		inputSource = yolo.NewRTSPInput(inputPath)
	} else if strings.HasPrefix(inputPath, "rtmp://") {
		inputSource = yolo.NewRTMPInput(inputPath)
	} else if inputPath == "screen" || inputPath == "desktop" {
		inputSource = yolo.NewScreenInput()
	} else if inputPath == "camera" || inputPath == "0" {
		inputSource = yolo.NewCameraInput("0")
	} else {
		// 默认为文件输入
		inputSource = yolo.NewFileInput(inputPath)
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
		boxColor:      "red",
		labelColor:    "white",
		lineWidth:     2,
		fontSize:      12,
		showFPS:       options.ShowFPS,
		stopChan:      make(chan bool),

		// 性能配置 - 默认平衡模式
		performanceMode: "balanced",
		frameSkip:       2,
		maxImageWidth:   800,
		maxImageHeight:  600,
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

	// 创建状态标签
	live.statusLabel = widget.NewLabel("准备自动播放...")
	live.fpsLabel = widget.NewLabel("FPS: 0")

	// 创建控制按钮
	playBtn := widget.NewButton("播放", live.startPlayback)
	stopBtn := widget.NewButton("停止", live.stopPlayback)

	// 创建设备信息标签
	deviceInfo := widget.NewLabel(fmt.Sprintf("设备: %s", live.inputSource.Path))

	// 创建布局
	controls := container.NewHBox(playBtn, stopBtn, live.statusLabel, live.fpsLabel)
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
	// 使用Vidio处理视频
	processor := yolo.NewVidioVideoProcessor(live.detector)

	// 根据性能模式设置参数
	switch live.performanceMode {
	case "fast":
		live.frameSkip = 3
		live.maxImageWidth = 640
		live.maxImageHeight = 480
	case "balanced":
		live.frameSkip = 2
		live.maxImageWidth = 800
		live.maxImageHeight = 600
	case "accurate":
		live.frameSkip = 1
		live.maxImageWidth = 1024
		live.maxImageHeight = 768
	}

	frameCount := 0

	// 使用输入源的FFmpeg参数
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

		// 使用fyne.Do在主线程中更新UI
		fyne.Do(func() {
			// 更新FPS显示
			if live.showFPS {
				live.fpsLabel.SetText(fmt.Sprintf("FPS: %.1f", live.fps))
			}

			// 如果有图像数据，显示它
			if result.Image != nil {
				// 在图像上绘制检测结果
				processedImage := live.drawDetectionsOnImage(result.Image, result.Detections)

				// 更新显示
				live.imageDisplay.Image = processedImage
				live.imageDisplay.Refresh()
			}

			// 更新状态
			live.statusLabel.SetText(fmt.Sprintf("帧: %d, 检测: %d", live.frameCount, len(result.Detections)))
		})

		// 控制播放速度 - 减少延迟
		time.Sleep(16 * time.Millisecond) // 约60 FPS
	})

	if err != nil {
		fyne.Do(func() {
			live.statusLabel.SetText(fmt.Sprintf("处理失败: %v", err))
		})
	}
}

// drawDetectionsOnImage 在图像上绘制检测结果
func (live *YOLOLiveWindow) drawDetectionsOnImage(img image.Image, detections []yolo.Detection) image.Image {
	// 性能优化：缩放图像以提高处理速度
	// 如果图像太大，先缩放到合适大小
	bounds := img.Bounds()
	if bounds.Dx() > live.maxImageWidth || bounds.Dy() > live.maxImageHeight {
		// 计算缩放比例
		scaleX := float64(live.maxImageWidth) / float64(bounds.Dx())
		scaleY := float64(live.maxImageHeight) / float64(bounds.Dy())
		scale := scaleX
		if scaleY < scaleX {
			scale = scaleY
		}

		// 缩放图像
		newWidth := int(float64(bounds.Dx()) * scale)
		newHeight := int(float64(bounds.Dy()) * scale)
		img = imaging.Resize(img, newWidth, newHeight, imaging.Lanczos)
		bounds = img.Bounds()
	}

	// 转换为RGBA
	result := image.NewRGBA(bounds)
	draw.Draw(result, bounds, img, bounds.Min, draw.Src)

	// 绘制检测框和标签
	for _, detection := range detections {
		if live.drawBoxes {
			live.drawBox(result, detection.Box, live.getColor(live.boxColor))
		}
		if live.drawLabels {
			live.drawLabel(result, detection.Class, detection.Score, detection.Box)
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

	// 绘制文本背景
	textWidth := len(label) * live.fontSize / 2
	textHeight := live.fontSize

	for i := 0; i < textWidth && x+i < bounds.Max.X; i++ {
		for j := 0; j < textHeight && y+j < bounds.Max.Y; j++ {
			img.Set(x+i, y+j, color.Black)
		}
	}

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
func (live *YOLOLiveWindow) Run() {
	live.window.ShowAndRun()
}

// LaunchFyneLiveWindow 启动Fyne实时窗口
func LaunchFyneLiveWindow(detector *yolo.YOLO, videoPath string, options *yolo.DetectionOptions) error {
	fmt.Printf("🎬 启动Fyne GUI窗口，视频: %s\n", videoPath)

	// 创建并运行GUI窗口
	liveWindow := NewYOLOLiveWindow(detector, videoPath, options)
	liveWindow.Run()

	return nil
}
