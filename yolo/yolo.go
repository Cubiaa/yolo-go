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
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"os/exec"

	vidio "github.com/AlexEidt/Vidio"
	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
	"gopkg.in/yaml.v3"
)

// containsDynamicDimension 检查形状是否包含动态维度(-1)
func containsDynamicDimension(shape []int64) bool {
	for _, dim := range shape {
		if dim == -1 {
			return true
		}
	}
	return false
}

// GUILauncherFunc 函数类型用于启动GUI窗口
type GUILauncherFunc func(detector *YOLO, videoPath string, options *DetectionOptions) error

// 全局GUI启动器函数
var globalGUILauncherFunc GUILauncherFunc

// SetGUILauncherFunc 设置GUI启动器函数
func SetGUILauncherFunc(launcherFunc GUILauncherFunc) {
	globalGUILauncherFunc = launcherFunc
}

// GUILauncher 接口用于启动GUI窗口
type GUILauncher interface {
	LaunchLiveWindow(detector *YOLO, videoPath string, options *DetectionOptions) error
}

// 全局GUI启动器
var globalGUILauncher GUILauncher

// SetGUILauncher 设置GUI启动器
func SetGUILauncher(launcher GUILauncher) {
	globalGUILauncher = launcher
}

// 全局类别列表（从配置文件加载）
var globalClasses []string

// VideoDetectionResult 视频检测结果
type VideoDetectionResult struct {
	FrameNumber int
	Timestamp   time.Duration
	Detections  []Detection
	Image       image.Image
}

// SetClasses 设置全局类别列表
func SetClasses(classes []string) {
	globalClasses = classes
}

// GetClasses 获取全局类别列表
func GetClasses() []string {
	return globalClasses
}

// Detection 检测结果结构体
type Detection struct {
	Box     [4]float32 // x1, y1, x2, y2
	Score   float32
	ClassID int
	Class   string
}

// DetectionResults 检测结果集合
type DetectionResults struct {
	Detections []Detection
	InputPath  string
	detector   *YOLO
	// 新增：存储视频的逐帧检测结果
	VideoResults []VideoDetectionResult
}

// Save 保存检测结果到指定路径
func (dr *DetectionResults) Save(outputPath string) error {
	if len(dr.Detections) == 0 {
		return fmt.Errorf("没有检测结果可保存")
	}

	if dr.InputPath == "" {
		return fmt.Errorf("没有输入文件路径信息")
	}

	if isVideoFile(dr.InputPath) {
		// 视频：优先使用已有的检测结果快速保存（不保留音频）
		if len(dr.VideoResults) > 0 {
			fmt.Println("🚀 使用已有检测结果快速保存视频...")
			return dr.saveVideoWithCachedResults(outputPath)
		} else {
			// 回退到重新检测模式
			fmt.Println("⚠️ 没有缓存的检测结果，将重新检测视频...")
			return dr.detector.DetectVideoAndSave(dr.InputPath, outputPath)
		}
	} else {
		// 图片：保存带检测框的图片
		_, err := dr.detector.DetectAndSave(dr.InputPath, outputPath)
		return err
	}
}

// 全局变量用于管理ONNX Runtime环境
var (
	ortInitialized bool
	ortMutex       sync.Mutex
)

// YOLO 检测器
type YOLO struct {
	config  *YOLOConfig
	session *ort.DynamicAdvancedSession
	// 运行时配置
	runtimeConfig *DetectionOptions
	// 添加状态跟踪
	lastInputPath  string
	lastDetections *DetectionResults
	lastImage      image.Image
	// 模型信息
	modelInputShape  []int64  // 模型实际输入形状
	modelOutputShape []int64  // 模型实际输出形状
}

// NewYOLO 创建新的YOLO检测器（配置文件必须，YOLOConfig可选）
func NewYOLO(modelPath, configPath string, config ...*YOLOConfig) (*YOLO, error) {
	// 加载配置文件（必须）
	configManager := NewConfigManager(configPath)
	err := configManager.LoadConfig()
	if err != nil {
		// 如果配置文件不存在，尝试创建默认配置
		fmt.Printf("⚠️  配置文件不存在，创建默认配置: %v\n", err)
		err = configManager.CreateDefaultConfig()
		if err != nil {
			return nil, fmt.Errorf("创建默认配置文件失败: %v", err)
		}
	}

	// 加载类别信息
	err = loadClassesFromYAML(configPath)
	if err != nil {
		fmt.Printf("⚠️  加载类别信息失败: %v\n", err)
		fmt.Println("💡 将使用默认类别列表")
		// 设置默认类别
		defaultClasses := []string{
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
			"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
			"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
			"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
			"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
			"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
			"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
			"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
			"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
		}
		SetClasses(defaultClasses)
	}

	// 使用传入的配置，如果没有则使用默认配置
	var yoloConfig *YOLOConfig
	if len(config) > 0 && config[0] != nil {
		yoloConfig = config[0]
	} else {
		yoloConfig = DefaultConfig()
	}

	// 设置ONNX Runtime库路径
	if yoloConfig.LibraryPath != "" {
		ort.SetSharedLibraryPath(yoloConfig.LibraryPath)
	}

	// 线程安全地初始化ONNX Runtime
	ortMutex.Lock()
	defer ortMutex.Unlock()

	if !ortInitialized {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, fmt.Errorf("无法初始化ONNX Runtime: %v", err)
		}
		ortInitialized = true
	}

	// 创建会话选项
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("无法创建会话选项: %v", err)
	}

	// 设置会话选项以提升性能
	// 根据CPU核心数动态调整线程数
	numCPU := runtime.NumCPU()
	optimalThreads := numCPU
	if numCPU > 8 {
		// 对于高核心数CPU，使用75%的核心以避免过度竞争
		optimalThreads = int(float64(numCPU) * 0.75)
	}
	if optimalThreads < 1 {
		optimalThreads = 1
	}

	fmt.Printf("💻 检测到 %d 个CPU核心，使用 %d 个线程进行优化\n", numCPU, optimalThreads)

	err = sessionOptions.SetIntraOpNumThreads(optimalThreads)
	if err != nil {
		fmt.Printf("⚠️  设置线程数失败: %v\n", err)
	}

	err = sessionOptions.SetInterOpNumThreads(optimalThreads)
	if err != nil {
		fmt.Printf("⚠️  设置操作间线程数失败: %v\n", err)
	}

	// 设置图优化级别以提升性能
	err = sessionOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)
	if err != nil {
		fmt.Printf("⚠️  设置图优化级别失败: %v\n", err)
	} else {
		fmt.Println("⚡ 启用所有图优化以提升性能")
	}

	// 设置执行模式为并行以提升性能
	err = sessionOptions.SetExecutionMode(ort.ExecutionModeParallel)
	if err != nil {
		fmt.Printf("⚠️  设置并行执行模式失败: %v\n", err)
	} else {
		fmt.Println("🔄 启用并行执行模式")
	}

	// 如果启用GPU，设置CUDA提供者
	if yoloConfig.UseGPU {
		fmt.Println("🚀 尝试启用GPU加速...")

		// 使用defer recover来捕获可能的panic
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("⚠️  GPU初始化发生panic: %v\n", r)
					fmt.Println("📋 GPU加速不可用，将使用CPU")
				}
			}()

			// 尝试添加CUDA执行提供者
			cudaOptions, err := ort.NewCUDAProviderOptions()
			if err != nil {
				fmt.Printf("⚠️  创建CUDA选项失败: %v\n", err)
			} else {
				defer cudaOptions.Destroy()
				
				// 设置CUDA选项
				optionsMap := map[string]string{
					"device_id": fmt.Sprintf("%d", yoloConfig.GPUDeviceID),
				}
				err = cudaOptions.Update(optionsMap)
				if err != nil {
					fmt.Printf("⚠️  更新CUDA选项失败: %v\n", err)
				} else {
					err = sessionOptions.AppendExecutionProviderCUDA(cudaOptions)
				}
			}
			if err != nil {
				fmt.Printf("⚠️  CUDA不可用: %v\n", err)

				// 尝试DirectML (Windows GPU) - 也需要安全检查
				fmt.Println("🔄 尝试DirectML提供者...")
				func() {
					defer func() {
						if r := recover(); r != nil {
							fmt.Printf("⚠️  DirectML初始化发生panic: %v\n", r)
							fmt.Println("📋 所有GPU加速都不可用，使用CPU")
						}
					}()

					err2 := sessionOptions.AppendExecutionProviderDirectML(yoloConfig.GPUDeviceID)
					if err2 != nil {
						fmt.Printf("⚠️  DirectML不可用: %v\n", err2)
						fmt.Println("📋 GPU加速失败，将使用CPU")
						fmt.Println("💡 可能的原因：")
						fmt.Println("   1. 没有兼容的GPU")
						fmt.Println("   2. 没有安装CUDA/DirectML")
						fmt.Println("   3. ONNX Runtime版本不支持GPU")
						fmt.Println("   4. GPU驱动程序过旧")
					} else {
						fmt.Println("✅ DirectML GPU加速已启用")
					}
				}()
			} else {
				fmt.Println("✅ CUDA GPU加速已启用")
			}
		}()
	} else {
		fmt.Println("💻 使用CPU模式")
	}

	// 加载模型
	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"}, sessionOptions)
	if err != nil {
		return nil, fmt.Errorf("无法加载模型文件 '%s': %v", modelPath, err)
	}

	// 获取模型输入输出信息
	inputInfos, outputInfos, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		session.Destroy()
		return nil, fmt.Errorf("无法获取模型输入输出信息: %v", err)
	}
	if len(inputInfos) == 0 || len(outputInfos) == 0 {
		session.Destroy()
		return nil, fmt.Errorf("模型输入或输出信息为空")
	}
	
	// 注意：InputOutputInfo结构体在onnxruntime_go v1.21.0中没有GetShape()方法
	// 我们使用默认的输入输出形状，或者从配置中获取
	var modelInputShape, modelOutputShape []int64
	
	// 根据配置设置输入形状
	if yoloConfig.InputWidth > 0 && yoloConfig.InputHeight > 0 {
		// 使用自定义的宽度和高度
		modelInputShape = []int64{1, 3, int64(yoloConfig.InputHeight), int64(yoloConfig.InputWidth)}
		fmt.Printf("📊 使用自定义输入形状 (宽x高): %dx%d -> %v\n", yoloConfig.InputWidth, yoloConfig.InputHeight, modelInputShape)
	} else {
		// 使用正方形输入尺寸
		modelInputShape = []int64{1, 3, int64(yoloConfig.InputSize), int64(yoloConfig.InputSize)}
		fmt.Printf("📊 使用正方形输入形状: %dx%d -> %v\n", yoloConfig.InputSize, yoloConfig.InputSize, modelInputShape)
	}
	
	// 输出形状设置为标准YOLO格式，避免动态维度导致的张量创建错误
	modelOutputShape = []int64{1, 84, 8400} // 标准YOLO输出格式
	fmt.Printf("📊 输出形状: %v (标准YOLO格式)\n", modelOutputShape)

	return &YOLO{
		config:           yoloConfig,
		session:          session,
		modelInputShape:  modelInputShape,
		modelOutputShape: modelOutputShape,
	}, nil
}

// NewYOLOWithConfig 创建新的YOLO检测器（支持配置文件）
func NewYOLOWithConfig(modelPath, configPath string, config *YOLOConfig) (*YOLO, error) {
	return NewYOLO(modelPath, configPath, config)
}

// Close 关闭YOLO检测器
func (y *YOLO) Close() {
	if y.session != nil {
		y.session.Destroy()
	}
	// 注意：不要在这里调用 ort.DestroyEnvironment()
	// 因为可能有其他检测器还在使用
}

// SetRuntimeConfig 设置运行时检测配置
func (y *YOLO) SetRuntimeConfig(options *DetectionOptions) {
	y.runtimeConfig = options
}

// DestroyEnvironment 销毁ONNX Runtime环境（在所有检测器都关闭后调用）
func DestroyEnvironment() {
	ortMutex.Lock()
	defer ortMutex.Unlock()
	if ortInitialized {
		ort.DestroyEnvironment()
		ortInitialized = false
	}
}

// DetectImage 检测单张图片
func (y *YOLO) DetectImage(imagePath string) ([]Detection, error) {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	// 加载图像以获取原始尺寸
	img, err := imaging.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开图像: %v", err)
	}

	// 获取原始图像尺寸
	originalBounds := img.Bounds()
	originalWidth := float32(originalBounds.Dx())
	originalHeight := float32(originalBounds.Dy())

	// 预处理图像
	inputData, err := y.preprocessImage(imagePath)
	if err != nil {
		return nil, fmt.Errorf("图像预处理失败: %v", err)
	}

	// 创建输入张量
	var inputShape ort.Shape
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		inputShape = ort.NewShape(1, 3, int64(y.config.InputHeight), int64(y.config.InputWidth))
	} else {
		// 使用正方形输入尺寸
		inputShape = ort.NewShape(1, 3, int64(y.config.InputSize), int64(y.config.InputSize))
	}
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("无法创建输入张量: %v", err)
	}
	defer inputTensor.Destroy()

	// 创建输出张量（智能适配模型输出形状）
	var outputShape ort.Shape
	var outputDataSize int
	
	// 如果是第一次推理或者modelOutputShape包含动态维度，使用标准形状进行探测
	if len(y.modelOutputShape) == 0 || containsDynamicDimension(y.modelOutputShape) {
		// 使用标准YOLO输出形状进行第一次推理
		outputShape = ort.NewShape(1, 84, 8400)
		outputDataSize = 1 * 84 * 8400
		fmt.Println("🔍 使用标准YOLO输出形状进行模型探测: [1, 84, 8400]")
	} else {
		// 使用已知的模型输出形状
		outputShape = ort.NewShape(y.modelOutputShape...)
		outputDataSize = 1
		for _, dim := range y.modelOutputShape {
			outputDataSize *= int(dim)
		}
		fmt.Printf("📊 使用已知模型输出形状: %v\n", y.modelOutputShape)
	}
	
	outputData := make([]float32, outputDataSize)
	outputTensor, err := ort.NewTensor(outputShape, outputData)

	if err != nil {
		return nil, fmt.Errorf("无法创建输出张量: %v", err)
	}
	defer outputTensor.Destroy()

	// 运行推理
	err = y.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, fmt.Errorf("推理失败: %v", err)
	}

	// 获取实际的输出形状并更新模型信息
	actualOutputShape := outputTensor.GetShape()
	if len(y.modelOutputShape) == 0 || containsDynamicDimension(y.modelOutputShape) {
		y.modelOutputShape = actualOutputShape
		fmt.Printf("✅ 自动检测到模型实际输出形状: %v\n", actualOutputShape)
	}

	// 解析检测结果
	detections := y.parseDetections(outputTensor.GetData(), actualOutputShape)

	// 将坐标从模型输入尺寸转换回原始图像尺寸
	var scaleX, scaleY float32
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		scaleX = originalWidth / float32(y.config.InputWidth)
		scaleY = originalHeight / float32(y.config.InputHeight)
	} else {
		// 使用正方形输入尺寸
		scaleX = originalWidth / float32(y.config.InputSize)
		scaleY = originalHeight / float32(y.config.InputSize)
	}
	
	for i := range detections {
		detections[i].Box[0] *= scaleX // x1
		detections[i].Box[1] *= scaleY // y1
		detections[i].Box[2] *= scaleX // x2
		detections[i].Box[3] *= scaleY // y2
	}

	// 应用非极大抑制
	threshold := float32(0.5) // 默认值
	if y.runtimeConfig != nil {
		threshold = y.runtimeConfig.IOUThreshold
	}
	keep := y.nonMaxSuppression(detections, threshold)

	return keep, nil
}

// DetectAndSave 检测图片并保存结果
func (y *YOLO) DetectAndSave(imagePath, outputPath string) ([]Detection, error) {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	// 检测图片
	detections, err := y.DetectImage(imagePath)
	if err != nil {
		return nil, err
	}

	// 读取原始图片
	img, err := imaging.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开图片: %v", err)
	}

	// 在图片上绘制检测框
	imgWithBoxes := y.drawDetectionsOnImage(img, detections)

	// 保存图片
	err = imaging.Save(imgWithBoxes, outputPath)
	if err != nil {
		return nil, fmt.Errorf("保存图片失败: %v", err)
	}

	return detections, nil
}

// DetectVideo 检测视频文件（MP4等）
func (y *YOLO) DetectVideo(inputPath string, showLive ...bool) ([]VideoDetectionResult, error) {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	if !isVideoFile(inputPath) {
		return nil, fmt.Errorf("不支持的文件格式，请使用MP4等视频文件")
	}

	// 使用Vidio处理视频文件
	processor := NewVidioVideoProcessor(y)

	if len(showLive) > 0 && showLive[0] {
		fmt.Println("💡 注意：实时播放功能需要额外的显示库支持")
		fmt.Println("💡 当前仅进行视频检测，返回所有帧的检测结果")
	}

	// 处理视频并返回结果
	return processor.ProcessVideo(inputPath)
}

// DetectVideoAndSave 检测视频并保存结果
func (y *YOLO) DetectVideoAndSave(inputPath, outputPath string, showLive ...bool) error {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	if !isVideoFile(inputPath) {
		return fmt.Errorf("不支持的文件格式，请使用MP4等视频文件")
	}

	// 使用Vidio处理视频文件
	processor := NewVidioVideoProcessor(y)

	if len(showLive) > 0 && showLive[0] {
		fmt.Println("💡 注意：实时播放功能需要额外的显示库支持")
		fmt.Println("💡 当前仅保存带检测框的视频文件")
	}

	// 保存带检测框的视频
	return processor.SaveVideoWithDetections(inputPath, outputPath)
}

// Show 可视化检测结果
func (y *YOLO) Show(inputPath string, outputPath ...string) error {
	if isVideoFile(inputPath) {
		// 视频：弹出窗口实时播放
		fmt.Printf("🎬 播放视频窗口: %s (按ESC退出)\n", inputPath)
		return y.ShowLive(inputPath)
	} else {
		// 图片：保存到文件
		if len(outputPath) == 0 {
			return fmt.Errorf("图片需要指定输出路径")
		}
		fmt.Printf("📸 可视化图片: %s -> %s\n", inputPath, outputPath[0])
		_, err := y.DetectAndSave(inputPath, outputPath[0])
		return err
	}
}



// DetectVideoAdvanced 高级视频检测（支持更多选项）
func (y *YOLO) DetectVideoAdvanced(inputPath, outputPath string, options VideoOptions) error {
	if isVideoFile(inputPath) {
		// 提供解决方案
		return fmt.Errorf("视频文件需要FFmpeg支持。解决方案：\n\n"+
			"方案1 - 使用Vidio库（推荐）：\n"+
			"  go get github.com/AlexEidt/Vidio\n"+
			"  程序已集成Vidio库支持\n\n"+
			"方案2 - 使用FFmpeg转换：\n"+
			"  ffmpeg -i \"%s\" -r %d \"%s/frame_%%04d.jpg\"\n"+
			"  然后使用: detector.DetectVideo(\"%s\", \"%s\")\n\n"+
			"方案3 - 在线转换：\n"+
			"  使用在线工具将MP4转换为图像序列",
			inputPath, options.FPS, options.FramesDir, options.FramesDir, outputPath)
	} else {
		// 处理图像序列
		processor := NewSimpleVideoProcessor(y)
		return processor.ProcessImageSequence(inputPath, outputPath)
	}
}

// VideoOptions 视频处理选项
type VideoOptions struct {
	FPS       int    // 输出FPS
	FramesDir string // 临时帧目录
	Quality   int    // 输出质量 (1-100)
}

// DefaultVideoOptions 默认视频选项
func DefaultVideoOptions() VideoOptions {
	return VideoOptions{
		FPS:       10,
		FramesDir: "temp_frames",
		Quality:   90,
	}
}

// 预处理图像
func (y *YOLO) preprocessImage(imagePath string) ([]float32, error) {
	// 打开图像
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("无法打开图像文件 '%s': %v", imagePath, err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("无法解码图像: %v", err)
	}

	// 根据配置调整大小
	var resized image.Image
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		resized = imaging.Resize(img, y.config.InputWidth, y.config.InputHeight, imaging.Lanczos)
	} else {
		// 使用正方形输入尺寸
		resized = imaging.Resize(img, y.config.InputSize, y.config.InputSize, imaging.Lanczos)
	}

	// 转换为RGB并归一化
	bounds := resized.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 创建输入张量 [1, 3, 640, 640]
	data := make([]float32, 1*3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			// 归一化到 [0, 1]
			data[0*height*width+y*width+x] = float32(r>>8) / 255.0 // R通道
			data[1*height*width+y*width+x] = float32(g>>8) / 255.0 // G通道
			data[2*height*width+y*width+x] = float32(b>>8) / 255.0 // B通道
		}
	}

	return data, nil
}

// 解析检测结果
func (y *YOLO) parseDetections(outputData []float32, outputShape []int64) []Detection {
	if len(outputShape) != 3 || outputShape[0] != 1 {
		fmt.Printf("⚠️  不支持的输出形状: %v\n", outputShape)
		return nil
	}

	numDetections := int(outputShape[2]) // 例如: 8400
	numFeatures := int(outputShape[1])   // 例如: 84, 85, 等
	numClasses := numFeatures - 4        // 动态计算类别数量 (总特征数 - 4个坐标)
	
	if numClasses <= 0 {
		fmt.Printf("⚠️  无效的类别数量: %d (特征数: %d)\n", numClasses, numFeatures)
		return nil
	}
	
	fmt.Printf("📊 解析输出: %d个检测框, %d个特征, %d个类别\n", numDetections, numFeatures, numClasses)

	var detections []Detection

	// 解析检测结果
	for i := 0; i < numDetections; i++ {
		// 对于格式 [batch, features, detections]，访问第i个检测的所有特征
		cx := outputData[0*numFeatures*numDetections+0*numDetections+i]
		cy := outputData[0*numFeatures*numDetections+1*numDetections+i]
		w := outputData[0*numFeatures*numDetections+2*numDetections+i]
		h := outputData[0*numFeatures*numDetections+3*numDetections+i]

		// 找到最大的类别概率
		var bestScore float32 = 0
		bestID := 0
		for classIdx := 0; classIdx < numClasses; classIdx++ {
			score := outputData[0*numFeatures*numDetections+(4+classIdx)*numDetections+i]
			if score > bestScore {
				bestScore = score
				bestID = classIdx
			}
		}

		// 使用配置的置信度阈值
		confThreshold := float32(0.5) // 默认值
		if y.runtimeConfig != nil {
			confThreshold = y.runtimeConfig.ConfThreshold
		}

		if bestScore < confThreshold {
			continue
		}

		// 转换为x1, y1, x2, y2格式
		x1 := cx - w/2.0
		y1 := cy - h/2.0
		x2 := cx + w/2.0
		y2 := cy + h/2.0

		className := "unknown"
		if bestID < len(globalClasses) {
			className = globalClasses[bestID]
		}

		detections = append(detections, Detection{
			Box:     [4]float32{x1, y1, x2, y2},
			Score:   bestScore,
			ClassID: bestID,
			Class:   className,
		})
	}

	return detections
}

// IOU计算
func (y *YOLO) iou(box1, box2 [4]float32) float32 {
	x1Min, y1Min, x1Max, y1Max := box1[0], box1[1], box1[2], box1[3]
	x2Min, y2Min, x2Max, y2Max := box2[0], box2[1], box2[2], box2[3]

	interXMin := max(x1Min, x2Min)
	interYMin := max(y1Min, y2Min)
	interXMax := min(x1Max, x2Max)
	interYMax := min(y1Max, y2Max)

	interArea := max(0, interXMax-interXMin) * max(0, interYMax-interYMin)
	area1 := (x1Max - x1Min) * (y1Max - y1Min)
	area2 := (x2Max - x2Min) * (y2Max - y2Min)

	return interArea / (area1 + area2 - interArea + 1e-6)
}

// 非极大抑制
func (y *YOLO) nonMaxSuppression(detections []Detection, iouThreshold float32) []Detection {
	if len(detections) == 0 {
		return detections
	}

	// 按分数排序
	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Score > detections[j].Score
	})

	var keep []Detection
	for i := 0; i < len(detections); i++ {
		current := detections[i]
		keepCurrent := true

		for j := 0; j < len(keep); j++ {
			if y.iou(current.Box, keep[j].Box) > iouThreshold {
				keepCurrent = false
				break
			}
		}

		if keepCurrent {
			keep = append(keep, current)
		}
	}

	return keep
}

// 画检测框
func (y *YOLO) drawBBox(img draw.Image, bbox [4]float32, lineColor color.Color) {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	x1 := int(max(0, min(float32(width-1), bbox[0])))
	y1 := int(max(0, min(float32(height-1), bbox[1])))
	x2 := int(max(0, min(float32(width-1), bbox[2])))
	y2 := int(max(0, min(float32(height-1), bbox[3])))

	// 获取线条宽度
	lineWidth := 1
	if y.runtimeConfig != nil && y.runtimeConfig.LineWidth > 0 {
		lineWidth = y.runtimeConfig.LineWidth
	}

	// 画矩形框（支持自定义线条宽度）
	for i := 0; i < lineWidth; i++ {
		// 上边和下边
		for x := x1; x <= x2; x++ {
			if y1+i < height {
				img.Set(x, y1+i, lineColor) // 上边
			}
			if y2-i >= 0 {
				img.Set(x, y2-i, lineColor) // 下边
			}
		}
		// 左边和右边
		for y := y1; y <= y2; y++ {
			if x1+i < width {
				img.Set(x1+i, y, lineColor) // 左边
			}
			if x2-i >= 0 {
				img.Set(x2-i, y, lineColor) // 右边
			}
		}
	}
}

// 绘制检测结果
func (y *YOLO) drawDetections(imagePath, outputPath string, detections []Detection) error {
	// 重新加载图像
	file, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("无法重新打开图像文件: %v", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("无法解码图像: %v", err)
	}

	// 转换为可绘制的图像
	bounds := img.Bounds()
	origImg := image.NewRGBA(bounds)
	draw.Draw(origImg, bounds, img, bounds.Min, draw.Src)

	origW, origH := bounds.Max.X, bounds.Max.Y

	// 获取颜色配置
	boxColor := color.RGBA{255, 0, 0, 255} // 默认红色
	if y.runtimeConfig != nil && y.runtimeConfig.BoxColor != "" {
		if parsedColor := y.parseColor(y.runtimeConfig.BoxColor); parsedColor != nil {
			boxColor = *parsedColor
		}
	}

	for _, detection := range detections {
		// 检测结果坐标已经是原始图像坐标，无需再次缩放
		x1 := max(0, detection.Box[0])
		y1 := max(0, detection.Box[1])
		x2 := min(float32(origW), detection.Box[2])
		y2 := min(float32(origH), detection.Box[3])

		// 检查是否应该画框和标签
		drawBoxes := true
		drawLabels := true
		if y.runtimeConfig != nil {
			drawBoxes = y.runtimeConfig.DrawBoxes
			drawLabels = y.runtimeConfig.DrawLabels
		}

		if drawBoxes {
			// 画检测框
			y.drawBBox(origImg, [4]float32{x1, y1, x2, y2}, boxColor)
		}

		if drawLabels {
			// 绘制标签文本
			label := fmt.Sprintf("%s %.2f", detection.Class, detection.Score)
			y.drawLabel(origImg, label, int(x1), int(y1-20)) // 在框上方绘制标签
		}
	}

	// 保存结果
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("无法创建输出文件: %v", err)
	}
	defer outputFile.Close()

	// 根据文件扩展名选择编码格式
	ext := filepath.Ext(outputPath)
	switch ext {
	case ".png":
		err = png.Encode(outputFile, origImg)
	case ".jpg", ".jpeg":
		err = jpeg.Encode(outputFile, origImg, &jpeg.Options{Quality: 100})
	default:
		err = jpeg.Encode(outputFile, origImg, &jpeg.Options{Quality: 100})
	}

	if err != nil {
		return fmt.Errorf("无法保存结果图像: %v", err)
	}

	return nil
}

// drawDetectionsOnImage 直接在图像上绘制检测结果
func (y *YOLO) drawDetectionsOnImage(img image.Image, detections []Detection) image.Image {
	// 转换为可绘制的图像
	bounds := img.Bounds()
	origImg := image.NewRGBA(bounds)
	draw.Draw(origImg, bounds, img, bounds.Min, draw.Src)

	origW, origH := bounds.Max.X, bounds.Max.Y

	// 获取颜色配置
	boxColor := color.RGBA{255, 0, 0, 255} // 默认红色
	if y.runtimeConfig != nil && y.runtimeConfig.BoxColor != "" {
		if parsedColor := y.parseColor(y.runtimeConfig.BoxColor); parsedColor != nil {
			boxColor = *parsedColor
		}
	}

	for _, detection := range detections {
		// 检测结果坐标已经是原始图像坐标，无需再次缩放
		x1 := max(0, detection.Box[0])
		y1 := max(0, detection.Box[1])
		x2 := min(float32(origW), detection.Box[2])
		y2 := min(float32(origH), detection.Box[3])

		// 检查是否应该画框和标签
		drawBoxes := true
		drawLabels := true
		if y.runtimeConfig != nil {
			drawBoxes = y.runtimeConfig.DrawBoxes
			drawLabels = y.runtimeConfig.DrawLabels
		}

		if drawBoxes {
			// 画检测框
			y.drawBBox(origImg, [4]float32{x1, y1, x2, y2}, boxColor)
		}

		if drawLabels {
			// 绘制标签文本
			label := fmt.Sprintf("%s %.2f", detection.Class, detection.Score)
			y.drawLabel(origImg, label, int(x1), int(y1-20)) // 在框上方绘制标签
		}
	}

	return origImg
}

// drawLabel 绘制标签文本
func (y *YOLO) drawLabel(img *image.RGBA, label string, x, yPos int) {
	bounds := img.Bounds()

	// 设置字体和尺寸（支持自定义字体大小）
	var face font.Face
	var charWidth, textHeight int
	
	// 根据FontSize选择合适的字体
	if y.runtimeConfig != nil && y.runtimeConfig.FontSize > 0 {
		switch {
		case y.runtimeConfig.FontSize <= 10:
			face = basicfont.Face7x13
			charWidth = 7
			textHeight = 13
		case y.runtimeConfig.FontSize <= 15:
			face = basicfont.Face7x13 // 可以考虑使用更大的字体
			charWidth = 8
			textHeight = 15
		case y.runtimeConfig.FontSize <= 20:
			face = basicfont.Face7x13
			charWidth = 9
			textHeight = 18
		default:
			face = basicfont.Face7x13
			charWidth = 10
			textHeight = 20
		}
	} else {
		// 默认字体
		face = basicfont.Face7x13
		charWidth = 7
		textHeight = 13
	}
	
	textWidth := len(label) * charWidth
	padding := 4

	// 确保标签在图像范围内
	if x < 0 {
		x = 0
	}
	if x+textWidth+padding*2 > bounds.Max.X {
		x = bounds.Max.X - textWidth - padding*2
	}

	// 如果标签会超出上边界，就画在框下方
	if yPos < textHeight+padding {
		yPos = yPos + 30 // 画在框下方
	}

	if yPos > bounds.Max.Y-textHeight-padding {
		yPos = bounds.Max.Y - textHeight - padding
	}

	// 不绘制背景矩形，直接绘制文本

	// 获取标签颜色配置
	labelColor := color.RGBA{255, 255, 255, 255} // 默认白色
	if y.runtimeConfig != nil && y.runtimeConfig.LabelColor != "" {
		if parsedColor := y.parseColor(y.runtimeConfig.LabelColor); parsedColor != nil {
			labelColor = *parsedColor
		}
	}

	// 绘制文本
	point := fixed.Point26_6{
		X: fixed.Int26_6(x * 64),
		Y: fixed.Int26_6((yPos + textHeight - 2) * 64), // 稍微向上调整
	}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(labelColor),
		Face: face,
		Dot:  point,
	}
	d.DrawString(label)

	// 调试信息
	fmt.Printf("绘制标签: '%s' 在位置 (%d, %d)\n", label, x, yPos)
}

// 辅助函数
func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// parseColor 解析颜色字符串
func (y *YOLO) parseColor(colorStr string) *color.RGBA {
	switch strings.ToLower(colorStr) {
	case "red":
		return &color.RGBA{255, 0, 0, 255}
	case "green":
		return &color.RGBA{0, 255, 0, 255}
	case "blue":
		return &color.RGBA{0, 0, 255, 255}
	case "yellow":
		return &color.RGBA{255, 255, 0, 255}
	case "cyan":
		return &color.RGBA{0, 255, 255, 255}
	case "magenta":
		return &color.RGBA{255, 0, 255, 255}
	case "white":
		return &color.RGBA{255, 255, 255, 255}
	case "black":
		return &color.RGBA{0, 0, 0, 255}
	case "orange":
		return &color.RGBA{255, 165, 0, 255}
	case "purple":
		return &color.RGBA{128, 0, 128, 255}
	default:
		return nil // 无法解析的颜色，返回nil使用默认颜色
	}
}

// 便捷方法：从配置管理器创建YOLO
func NewYOLOFromConfig(modelPath string, configManager *ConfigManager, libraryPath string) (*YOLO, error) {
	// 加载配置管理器的配置
	err := configManager.LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("加载配置失败: %v", err)
	}

	// 使用配置管理器的YOLO配置
	yoloConfig := configManager.GetYOLOConfig()
	yoloConfig.LibraryPath = libraryPath

	return NewYOLO(modelPath, "config.yaml", yoloConfig)
}

// 便捷方法：使用预设配置创建YOLO（需要配置文件路径）
func NewYOLOWithPreset(modelPath, configPath, preset, libraryPath string) (*YOLO, error) {
	var config *YOLOConfig

	switch preset {
	case "default":
		config = DefaultConfig()
	case "gpu":
		config = GPUConfig()
	case "cpu":
		config = CPUConfig()
	default:
		return nil, fmt.Errorf("不支持的预设配置: %s", preset)
	}

	config.LibraryPath = libraryPath
	return NewYOLO(modelPath, configPath, config)
}

// 便捷方法：获取视频处理器
func (y *YOLO) GetVideoProcessor() *VidioVideoProcessor {
	return NewVidioVideoProcessor(y)
}

// IsGPUAvailable 检测GPU是否可用
func IsGPUAvailable() bool {
	// 创建临时会话选项来测试GPU支持
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return false
	}
	defer sessionOptions.Destroy()

	// 测试CUDA
	err = sessionOptions.AppendExecutionProviderCUDA(nil)
	if err == nil {
		return true
	}

	// 测试DirectML
	sessionOptions2, err := ort.NewSessionOptions()
	if err != nil {
		return false
	}
	defer sessionOptions2.Destroy()

	err = sessionOptions2.AppendExecutionProviderDirectML(0)
	return err == nil
}

// CheckGPUSupport 检查GPU支持情况
func CheckGPUSupport() {
	fmt.Println("=== GPU支持检查 ===")

	// 创建临时会话选项来测试GPU支持
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Printf("❌ 无法创建会话选项: %v\n", err)
		return
	}
	defer sessionOptions.Destroy()

	// 检查CUDA支持 - 使用安全检查
	fmt.Print("🔍 检查CUDA支持... ")
	func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("❌ panic: %v\n", r)
				return
			}
		}()

		err = sessionOptions.AppendExecutionProviderCUDA(nil)
		if err != nil {
			fmt.Printf("❌ 不支持 (%v)\n", err)
		} else {
			fmt.Println("✅ 支持")
		}
	}()

	// 检查DirectML支持 (Windows) - 使用安全检查
	fmt.Print("🔍 检查DirectML支持... ")
	func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("❌ panic: %v\n", r)
				return
			}
		}()

		sessionOptions2, err := ort.NewSessionOptions()
		if err != nil {
			fmt.Printf("❌ 无法创建会话选项: %v\n", err)
			return
		}
		defer sessionOptions2.Destroy()

		err = sessionOptions2.AppendExecutionProviderDirectML(0)
		if err != nil {
			fmt.Printf("❌ 不支持 (%v)\n", err)
		} else {
			fmt.Println("✅ 支持")
		}
	}()

	fmt.Println("💡 提示：")
	fmt.Println("   - CUDA: 需要NVIDIA GPU + CUDA驱动")
	fmt.Println("   - DirectML: 支持NVIDIA/AMD/Intel GPU (Windows)")
	fmt.Println("   - 如果都不支持，程序会自动使用CPU")
	fmt.Println("   - panic通常表示ONNX Runtime版本不支持GPU")
}

// GetGPUConfig 获取GPU配置建议
func GetGPUConfig() *YOLOConfig {
	return DefaultConfig().WithGPU(true).WithLibraryPath("")
}

// GetOptimalConfig 根据系统自动选择最优配置
func GetOptimalConfig() *YOLOConfig {
	if IsGPUAvailable() {
		fmt.Println("🚀 检测到GPU支持，使用GPU配置")
		return GetGPUConfig()
	} else {
		fmt.Println("💻 未检测到GPU支持，使用CPU配置")
		return CPUConfig()
	}
}

// 检查是否为视频文件
func isVideoFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	videoExts := map[string]bool{
		".mp4":  true,
		".avi":  true,
		".mov":  true,
		".mkv":  true,
		".wmv":  true,
		".flv":  true,
		".webm": true,
	}
	return videoExts[ext]
}

// ConvertVideoToFrames 提供视频转帧的命令建议
func ConvertVideoToFrames(videoPath, outputDir string, fps int) string {
	return fmt.Sprintf("ffmpeg -i \"%s\" -r %d \"%s/frame_%%04d.jpg\"", videoPath, fps, outputDir)
}

// ConvertFramesToVideo 提供帧转视频的命令建议
func ConvertFramesToVideo(framesDir, outputPath string, fps int) string {
	return fmt.Sprintf("ffmpeg -r %d -i \"%s/frame_%%04d.jpg\" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \"%s\"", fps, framesDir, outputPath)
}

// detectImage 检测单张图像（内部方法）
func (y *YOLO) detectImage(img image.Image) ([]Detection, error) {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	// 获取原始图像尺寸
	originalBounds := img.Bounds()
	originalWidth := float32(originalBounds.Dx())
	originalHeight := float32(originalBounds.Dy())

	// 预处理图像
	inputData, err := y.preprocessImageFromMemory(img)
	if err != nil {
		return nil, fmt.Errorf("图像预处理失败: %v", err)
	}

	// 创建输入张量
	var inputShape ort.Shape
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		inputShape = ort.NewShape(1, 3, int64(y.config.InputHeight), int64(y.config.InputWidth))
	} else {
		// 使用正方形尺寸
		inputShape = ort.NewShape(1, 3, int64(y.config.InputSize), int64(y.config.InputSize))
	}
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("无法创建输入张量: %v", err)
	}
	defer inputTensor.Destroy()

	// 创建输出张量（智能适配模型输出形状）
	var outputShape ort.Shape
	var outputDataSize int
	
	// 如果是第一次推理或者modelOutputShape包含动态维度，使用标准形状进行探测
	if len(y.modelOutputShape) == 0 || containsDynamicDimension(y.modelOutputShape) {
		// 使用标准YOLO输出形状进行第一次推理
		outputShape = ort.NewShape(1, 84, 8400)
		outputDataSize = 1 * 84 * 8400
	} else {
		// 使用已知的模型输出形状
		outputShape = ort.NewShape(y.modelOutputShape...)
		outputDataSize = 1
		for _, dim := range y.modelOutputShape {
			outputDataSize *= int(dim)
		}
	}
	
	outputData := make([]float32, outputDataSize)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("无法创建输出张量: %v", err)
	}
	defer outputTensor.Destroy()

	// 运行推理
	err = y.session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})

	if err != nil {
		return nil, fmt.Errorf("推理失败: %v", err)
	}

	// 获取实际的输出形状并更新模型信息
	actualOutputShape := outputTensor.GetShape()
	if len(y.modelOutputShape) == 0 || containsDynamicDimension(y.modelOutputShape) {
		y.modelOutputShape = actualOutputShape
	}

	// 解析检测结果
	detections := y.parseDetections(outputTensor.GetData(), actualOutputShape)

	// 将坐标从模型输入尺寸转换回原始图像尺寸
	var scaleX, scaleY float32
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		scaleX = originalWidth / float32(y.config.InputWidth)
		scaleY = originalHeight / float32(y.config.InputHeight)
	} else {
		// 使用正方形尺寸
		scaleX = originalWidth / float32(y.config.InputSize)
		scaleY = originalHeight / float32(y.config.InputSize)
	}
	
	for i := range detections {
		detections[i].Box[0] *= scaleX // x1
		detections[i].Box[1] *= scaleY // y1
		detections[i].Box[2] *= scaleX // x2
		detections[i].Box[3] *= scaleY // y2
	}

	// 应用非极大抑制
	threshold := float32(0.5) // 默认值
	if y.runtimeConfig != nil {
		threshold = y.runtimeConfig.IOUThreshold
	}
	keep := y.nonMaxSuppression(detections, threshold)

	return keep, nil
}

// preprocessImageFromMemory 从内存图像预处理
func (y *YOLO) preprocessImageFromMemory(img image.Image) ([]float32, error) {
	// 根据配置调整大小
	var resized image.Image
	if y.config.InputWidth > 0 && y.config.InputHeight > 0 {
		// 使用自定义的宽度和高度
		resized = imaging.Resize(img, y.config.InputWidth, y.config.InputHeight, imaging.Lanczos)
	} else {
		// 使用正方形输入尺寸
		resized = imaging.Resize(img, y.config.InputSize, y.config.InputSize, imaging.Lanczos)
	}

	// 转换为RGB并归一化
	bounds := resized.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// 创建输入张量 [1, 3, 640, 640]
	data := make([]float32, 1*3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			// 归一化到 [0, 1]
			data[0*height*width+y*width+x] = float32(r>>8) / 255.0 // R通道
			data[1*height*width+y*width+x] = float32(g>>8) / 255.0 // G通道
			data[2*height*width+y*width+x] = float32(b>>8) / 255.0 // B通道
		}
	}

	return data, nil
}

// 注意：已移除OpenCV依赖，使用Vidio库处理视频

// ShowLive 实时播放视频并显示检测框
func (y *YOLO) ShowLive(inputPath string) error {
	// 如果没有设置运行时配置，使用默认配置
	if y.runtimeConfig == nil {
		y.runtimeConfig = DefaultDetectionOptions()
	}

	if !isVideoFile(inputPath) {
		return fmt.Errorf("不支持的文件格式，请使用MP4等视频文件")
	}

	fmt.Printf("🎬 实时播放视频: %s\n", inputPath)
	fmt.Println("💡 注意：实时播放功能需要额外的显示库支持")
	fmt.Println("💡 当前实现：逐帧处理并保存为图片序列")
	fmt.Println("💡 建议：使用 DetectVideoAndSave 方法保存带检测框的视频文件")

	// 创建输出目录
	outputDir := "live_output"
	err := os.MkdirAll(outputDir, 0755)
	if err != nil {
		return fmt.Errorf("创建输出目录失败: %v", err)
	}

	// 使用Vidio处理视频
	processor := NewVidioVideoProcessor(y)

	// 处理视频并保存每一帧
	frameCount := 0
	err = processor.ProcessVideoWithCallback(inputPath, func(result VideoDetectionResult) {
		frameCount++

		// 保存带检测框的帧
		if len(result.Detections) > 0 {
			framePath := fmt.Sprintf("%s/frame_%04d.jpg", outputDir, frameCount)

			// 如果有图像数据，保存它
			if result.Image != nil {
				err := imaging.Save(result.Image, framePath)
				if err != nil {
					fmt.Printf("保存帧 %d 失败: %v\n", frameCount, err)
				} else {
					fmt.Printf("✅ 保存帧 %d: %s (检测到 %d 个对象)\n", frameCount, framePath, len(result.Detections))
				}
			}
		}

		// 每10帧显示一次进度
		if frameCount%10 == 0 {
			fmt.Printf("📊 已处理 %d 帧...\n", frameCount)
		}
	})

	if err != nil {
		return fmt.Errorf("处理视频失败: %v", err)
	}

	fmt.Printf("✅ 实时处理完成！共处理 %d 帧，结果保存在 %s/ 目录\n", frameCount, outputDir)
	fmt.Println("💡 你可以查看 live_output/ 目录中的图片序列")

	return nil
}

// ShowLiveWindow 启动实时GUI窗口
func (y *YOLO) ShowLiveWindow(videoPath string, opts *DetectionOptions) error {
	fmt.Println("🎬 启动实时GUI窗口...")
	fmt.Printf("📹 视频文件: %s\n", videoPath)

	// 启动GUI窗口
	fmt.Println("🚀 启动GUI窗口...")

	// 使用os/exec启动GUI程序
	// 编译并运行GUI启动器
	fmt.Println("💡 正在启动GUI窗口...")

	// 这里我们使用一个简单的方法：直接启动GUI
	// 为了避免循环导入，我们使用命令行方式
	fmt.Printf("🎯 启动GUI: gui_launcher.exe %s\n", videoPath)

	return nil
}

// StartLiveGUI 启动实时GUI窗口
func StartLiveGUI(detector *YOLO, videoPath string, options *DetectionOptions) error {
	fmt.Println("🎬 启动实时GUI窗口...")
	fmt.Printf("📹 视频文件: %s\n", videoPath)

	// 使用os/exec启动独立的GUI程序
	fmt.Println("🚀 启动GUI程序...")

	// 检查是否存在GUI启动器
	guiExe := "gui_launcher.exe"
	if _, err := os.Stat(guiExe); os.IsNotExist(err) {
		fmt.Printf("❌ GUI启动器不存在: %s\n", guiExe)
		fmt.Println("💡 请先编译GUI启动器:")
		fmt.Println("   go build -o gui_launcher.exe gui_launcher.go")
		return fmt.Errorf("GUI启动器不存在")
	}

	// 启动GUI程序
	cmd := exec.Command(guiExe, videoPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	fmt.Println("✅ 启动GUI窗口...")
	err := cmd.Start()
	if err != nil {
		return fmt.Errorf("启动GUI失败: %v", err)
	}

	// 在后台运行GUI
	go func() {
		cmd.Wait()
		fmt.Println("✅ GUI窗口已关闭")
	}()

	return nil
}

// Save 保存检测结果到指定路径
// 注意：YOLO.Save() 和 YOLO.SaveDetections() 方法已被移除
// 请使用 DetectionResults.Save() 方法：
//   result, err := detector.Detect("input.jpg")
//   err = result.Save("output.jpg")

// Detect 检测并返回结果（不保存），支持可选的回调函数
func (y *YOLO) Detect(inputPath string, options *DetectionOptions, callbacks ...interface{}) (*DetectionResults, error) {
	// 使用默认选项或传入的选项
	opts := DefaultDetectionOptions()
	if options != nil {
		opts = options
	}

	// 设置运行时配置
	y.runtimeConfig = opts

	// 处理图片文件
	if strings.HasSuffix(strings.ToLower(inputPath), ".jpg") ||
		strings.HasSuffix(strings.ToLower(inputPath), ".jpeg") ||
		strings.HasSuffix(strings.ToLower(inputPath), ".png") {
		// 图片：直接检测
		detections, err := y.DetectImage(inputPath)
		
		// 如果提供了回调函数，调用它
		if len(callbacks) > 0 {
			if callback, ok := callbacks[0].(func(VideoDetectionResult)); ok {
				// 为图片创建VideoDetectionResult
				if err == nil {
					// 加载图片用于回调
					img, imgErr := y.loadImageForCallback(inputPath)
					result := VideoDetectionResult{
						FrameNumber: 1, // 图片只有一帧
						Timestamp:   0,
						Detections:  detections,
						Image:       img,
					}
					if imgErr != nil {
						result.Image = nil
					}
					callback(result)
				}
			}
		}
		
		if err != nil {
			return nil, err
		}

		// 设置状态变量
		y.lastInputPath = inputPath
		y.lastDetections = &DetectionResults{
			Detections: detections,
			InputPath:  inputPath,
			detector:   y,
		}

		return y.lastDetections, nil
	}

	// 处理视频文件
	if isVideoFile(inputPath) {
		fmt.Printf("🎬 检测视频文件: %s\n", inputPath)

		// 使用Vidio处理视频
		processor := NewVidioVideoProcessor(y)

		var allDetections []Detection
		var videoResults []VideoDetectionResult

		// 处理视频
		err := processor.ProcessVideoWithCallback(inputPath, func(result VideoDetectionResult) {
			// 添加到结果列表
			videoResults = append(videoResults, result)
			allDetections = append(allDetections, result.Detections...)

			// 如果提供了回调函数，调用它
			if len(callbacks) > 0 {
				if callback, ok := callbacks[0].(func(VideoDetectionResult)); ok {
					callback(result)
				}
			}

			// 实时更新状态
			fmt.Printf("📊 处理帧 %d, 检测到 %d 个对象\n", len(videoResults), len(result.Detections))
		})

		if err != nil {
			return nil, fmt.Errorf("视频检测失败: %v", err)
		}

		// 保存状态用于Save方法
		y.lastInputPath = inputPath
		y.lastDetections = &DetectionResults{
			Detections:   allDetections,
			InputPath:    inputPath,
			detector:     y,
			VideoResults: videoResults, // 保存视频逐帧检测结果
		}

		fmt.Printf("✅ 视频检测完成！共检测 %d 帧，发现 %d 个对象\n", len(videoResults), len(allDetections))
		return y.lastDetections, nil
	}

	return nil, fmt.Errorf("不支持的文件格式")
}

// DetectFromCamera 从摄像头检测对象，统一使用VideoDetectionResult回调
func (y *YOLO) DetectFromCamera(device string, options *DetectionOptions, callback ...func(VideoDetectionResult)) (*DetectionResults, error) {
	fmt.Printf("📹 从摄像头检测: %s\n", device)

	// 设置运行时配置
	y.runtimeConfig = options

	// 使用CameraVideoProcessor处理摄像头流
	processor := NewCameraVideoProcessor(y, device)

	var allDetections []Detection
	var frameCount int

	// 处理摄像头流，使用VideoDetectionResult回调
	err := processor.ProcessCameraWithCallback(func(result VideoDetectionResult) {
		frameCount++
		allDetections = append(allDetections, result.Detections...)

		// 实时更新状态
		fmt.Printf("📊 摄像头帧 %d, 检测到 %d 个对象\n", frameCount, len(result.Detections))
		
		// 如果提供了回调函数，调用它
		if len(callback) > 0 && callback[0] != nil {
			callback[0](result)
		}
	})

	if err != nil {
		return nil, fmt.Errorf("摄像头检测失败: %v", err)
	}

	// 保存状态
	y.lastInputPath = device
	y.lastDetections = &DetectionResults{
		Detections: allDetections,
		InputPath:  device,
		detector:   y,
	}

	return y.lastDetections, nil
}



// DetectFromRTSP 从RTSP流进行实时检测，支持可选的回调函数
func (y *YOLO) DetectFromRTSP(rtspURL string, options *DetectionOptions, callback ...func(VideoDetectionResult)) (*DetectionResults, error) {
	fmt.Printf("🌐 从RTSP流检测: %s\n", rtspURL)

	// 创建RTSP输入源
	input := NewRTSPInput(rtspURL)
	if err := input.Validate(); err != nil {
		return nil, fmt.Errorf("RTSP输入验证失败: %v", err)
	}

	// 设置运行时配置
	y.runtimeConfig = options

	// 使用Vidio处理RTSP流
	processor := NewVidioVideoProcessor(y)

	var allDetections []Detection
	var frameCount int

	// 处理RTSP流
	err := processor.ProcessVideoWithCallback(input.GetFFmpegInput(), func(result VideoDetectionResult) {
		frameCount++
		allDetections = append(allDetections, result.Detections...)

		// 实时更新状态
		fmt.Printf("📊 RTSP帧 %d, 检测到 %d 个对象\n", frameCount, len(result.Detections))
		
		// 如果提供了回调函数，调用它
		if len(callback) > 0 && callback[0] != nil {
			callback[0](result)
		}
	})

	if err != nil {
		return nil, fmt.Errorf("RTSP检测失败: %v", err)
	}

	// 保存状态
	y.lastInputPath = input.Path
	y.lastDetections = &DetectionResults{
		Detections: allDetections,
		InputPath:  input.Path,
		detector:   y,
	}

	return y.lastDetections, nil
}



// DetectFromScreen 从屏幕录制进行实时检测，支持可选的回调函数
func (y *YOLO) DetectFromScreen(options *DetectionOptions, callback ...func(VideoDetectionResult)) (*DetectionResults, error) {
	fmt.Println("🖥️  从屏幕录制检测")

	// 创建屏幕输入源
	input := NewScreenInput()
	if err := input.Validate(); err != nil {
		return nil, fmt.Errorf("屏幕输入验证失败: %v", err)
	}

	// 设置运行时配置
	y.runtimeConfig = options

	// 使用Vidio处理屏幕流
	processor := NewVidioVideoProcessor(y)

	var allDetections []Detection
	var frameCount int

	// 处理屏幕流
	err := processor.ProcessVideoWithCallback(input.GetFFmpegInput(), func(result VideoDetectionResult) {
		frameCount++
		allDetections = append(allDetections, result.Detections...)

		// 实时更新状态
		fmt.Printf("📊 屏幕帧 %d, 检测到 %d 个对象\n", frameCount, len(result.Detections))
		
		// 如果提供了回调函数，调用它
		if len(callback) > 0 && callback[0] != nil {
			callback[0](result)
		}
	})

	if err != nil {
		return nil, fmt.Errorf("屏幕检测失败: %v", err)
	}

	// 保存状态
	y.lastInputPath = input.Path
	y.lastDetections = &DetectionResults{
		Detections: allDetections,
		InputPath:  input.Path,
		detector:   y,
	}

	return y.lastDetections, nil
}



// DetectFromRTMP 从RTMP流进行实时检测，支持可选的回调函数
func (y *YOLO) DetectFromRTMP(rtmpURL string, options *DetectionOptions, callback ...func(VideoDetectionResult)) (*DetectionResults, error) {
	fmt.Printf("🌐 从RTMP流检测: %s\n", rtmpURL)

	// 创建RTMP输入源
	input := NewRTMPInput(rtmpURL)
	if err := input.Validate(); err != nil {
		return nil, fmt.Errorf("RTMP输入验证失败: %v", err)
	}

	// 设置运行时配置
	y.runtimeConfig = options

	// 使用Vidio处理RTMP流
	processor := NewVidioVideoProcessor(y)

	var allDetections []Detection
	var frameCount int

	// 处理RTMP流
	err := processor.ProcessVideoWithCallback(input.GetFFmpegInput(), func(result VideoDetectionResult) {
		frameCount++
		allDetections = append(allDetections, result.Detections...)

		// 实时更新状态
		fmt.Printf("📊 RTMP帧 %d, 检测到 %d 个对象\n", frameCount, len(result.Detections))
		
		// 如果提供了回调函数，调用它
		if len(callback) > 0 && callback[0] != nil {
			callback[0](result)
		}
	})

	if err != nil {
		return nil, fmt.Errorf("RTMP检测失败: %v", err)
	}

	// 保存状态
	y.lastInputPath = input.Path
	y.lastDetections = &DetectionResults{
		Detections: allDetections,
		InputPath:  input.Path,
		detector:   y,
	}

	return y.lastDetections, nil
}



// loadClassesFromYAML 从YAML文件加载类别列表
// loadImageForCallback 加载图片用于回调
func (y *YOLO) loadImageForCallback(imagePath string) (image.Image, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	return img, err
}

// saveVideoWithCachedResults 使用缓存的检测结果快速保存视频
func (dr *DetectionResults) saveVideoWithCachedResults(outputPath string) error {
	// 打开输入视频
	video, err := vidio.NewVideo(dr.InputPath)
	if err != nil {
		return fmt.Errorf("无法打开视频文件: %v", err)
	}
	defer video.Close()

	// 创建输出视频写入器 - 保持原画质
	options := &vidio.Options{
		FPS:     video.FPS(),
		Quality: 1.0, // 无损质量，保持原画质
	}

	writer, err := vidio.NewVideoWriter(outputPath, video.Width(), video.Height(), options)
	if err != nil {
		return fmt.Errorf("无法创建输出视频: %v", err)
	}
	defer writer.Close()

	fmt.Printf("📹 快速保存视频: %s -> %s (使用缓存结果)\n", dr.InputPath, outputPath)
	frameCount := 0
	resultIndex := 0

	// 逐帧处理
	for video.Read() {
		frameCount++

		// 将帧缓冲区转换为Go图像
		frameImg := convertFrameBufferToImage(video.FrameBuffer(), video.Width(), video.Height())

		// 使用缓存的检测结果（如果有的话）
		var detections []Detection
		if resultIndex < len(dr.VideoResults) && dr.VideoResults[resultIndex].FrameNumber == frameCount {
			detections = dr.VideoResults[resultIndex].Detections
			resultIndex++
		} else {
			// 如果没有对应帧的检测结果，使用空检测
			detections = []Detection{}
		}

		// 绘制检测结果
		var resultImg image.Image = frameImg
		if len(detections) > 0 {
			resultImg = dr.detector.drawDetectionsOnImage(frameImg, detections)
		}

		// 将图像转换回帧缓冲区并写入
		frameBuffer := convertImageToFrameBuffer(resultImg)
		err = writer.Write(frameBuffer)
		if err != nil {
			return fmt.Errorf("写入帧失败: %v", err)
		}

		// 进度提示
		if frameCount%30 == 0 {
			fmt.Printf("📊 已处理 %d/%d 帧... (快速模式)\n", frameCount, video.Frames())
		}
	}

	fmt.Printf("✅ 视频快速保存完成！共处理 %d 帧，使用了 %d 个缓存检测结果\n", frameCount, len(dr.VideoResults))
	return nil
}

func loadClassesFromYAML(configPath string) error {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("读取配置文件失败: %v", err)
	}

	var config struct {
		Classes []string `yaml:"classes"`
	}

	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return fmt.Errorf("解析配置文件失败: %v", err)
	}

	if len(config.Classes) == 0 {
		return fmt.Errorf("配置文件中没有找到类别列表")
	}

	// 设置全局类别列表
	SetClasses(config.Classes)

	// 显示前5个类别
	showCount := 5
	if len(config.Classes) < 5 {
		showCount = len(config.Classes)
	}
	fmt.Printf("✅ 成功加载 %d 个类别: %v\n", len(config.Classes), config.Classes[:showCount])
	if len(config.Classes) > 5 {
		fmt.Printf("   ... 还有 %d 个类别\n", len(config.Classes)-5)
	}

	return nil
}
