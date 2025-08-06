package yolo

import (
	"fmt"
	"path/filepath"
	"strings"
)

// YOLOConfig YOLO检测器配置（检测器级别 - 创建时设置）
type YOLOConfig struct {
	InputSize   int    // 输入尺寸（正方形时使用）
	InputWidth  int    // 输入宽度（非正方形时使用）
	InputHeight int    // 输入高度（非正方形时使用）
	UseGPU      bool   // 是否使用GPU
	GPUDeviceID int    // GPU设备ID（默认0，仅在UseGPU=true时有效）
	LibraryPath string // ONNX Runtime库路径
	AutoCreateConfig bool // 是否自动创建配置文件（默认false）
}

// DetectionOptions 检测选项
type DetectionOptions struct {
	ConfThreshold float32 // 置信度阈值
	IOUThreshold  float32 // IOU阈值
	DrawBoxes     bool    // 是否绘制检测框
	DrawLabels    bool    // 是否绘制标签
	ShowFPS       bool    // 是否显示FPS
	BoxColor      string  // 检测框颜色
	LabelColor    string  // 标签颜色
	LineWidth     int     // 线条宽度
	FontSize      int     // 字体大小
}

// DefaultConfig 返回默认极限性能配置（检测器级别）
// 现在集成了自动模型检测功能
func DefaultConfig() *YOLOConfig {
	// 自动检测硬件并选择极限性能配置
	if IsGPUAvailable() {
		fmt.Println("🚀 默认启用GPU极限性能模式（含智能模型适配）")
		return &YOLOConfig{
			InputSize:   640, // 默认尺寸，将在模型加载时自动调整
			UseGPU:      true,
			LibraryPath: "",
		}
	} else {
		fmt.Println("💻 默认启用CPU极限性能模式（含智能模型适配）")
		return &YOLOConfig{
			InputSize:   640, // 默认尺寸，将在模型加载时自动调整
			UseGPU:      false,
			LibraryPath: "",
		}
	}
}

// DefaultConfigWithModelPath 根据模型路径返回智能配置（推荐使用）
func DefaultConfigWithModelPath(modelPath string) *YOLOConfig {
	// 从模型文件中检测输入尺寸
	inputSize := detectModelInputSize(modelPath)
	if inputSize == 0 {
		// 如果检测失败，使用默认值
		inputSize = 640
		fmt.Printf("⚠️  无法检测模型输入尺寸，使用默认值: %d\n", inputSize)
	} else {
		fmt.Printf("✅ 自动检测到模型输入尺寸: %d\n", inputSize)
	}
	
	if IsGPUAvailable() {
		fmt.Printf("🚀 GPU极限性能模式 - 输入尺寸: %d\n", inputSize)
		return &YOLOConfig{
			InputSize:   inputSize,
			UseGPU:      true,
			LibraryPath: "",
		}
	} else {
		fmt.Printf("💻 CPU极限性能模式 - 输入尺寸: %d\n", inputSize)
		return &YOLOConfig{
			InputSize:   inputSize,
			UseGPU:      false,
			LibraryPath: "",
		}
	}
}

// WithInputSize 设置输入尺寸（正方形）
func (c *YOLOConfig) WithInputSize(size int) *YOLOConfig {
	c.InputSize = size
	c.InputWidth = 0  // 清除非正方形设置
	c.InputHeight = 0 // 清除非正方形设置
	return c
}

// WithInputDimensions 设置输入尺寸（宽度和高度）
func (c *YOLOConfig) WithInputDimensions(width, height int) *YOLOConfig {
	c.InputWidth = width
	c.InputHeight = height
	c.InputSize = 0 // 清除正方形设置
	return c
}

// WithAutoCreateConfig 设置是否自动创建配置文件
func (c *YOLOConfig) WithAutoCreateConfig(autoCreate bool) *YOLOConfig {
	c.AutoCreateConfig = autoCreate
	return c
}

// AutoDetectInputSizeConfig 自动检测模型输入尺寸的配置
func AutoDetectInputSizeConfig(modelPath string) *YOLOConfig {
	// 尝试从模型文件中检测输入尺寸
	inputSize := detectModelInputSize(modelPath)
	if inputSize == 0 {
		// 如果检测失败，使用默认值
		inputSize = 640
		fmt.Printf("⚠️  无法检测模型输入尺寸，使用默认值: %d\n", inputSize)
	} else {
		fmt.Printf("✅ 自动检测到模型输入尺寸: %d\n", inputSize)
	}
	
	if IsGPUAvailable() {
		fmt.Printf("🚀 GPU模式 - 输入尺寸: %d\n", inputSize)
		return &YOLOConfig{
			InputSize:   inputSize,
			UseGPU:      true,
			LibraryPath: "",
		}
	} else {
		fmt.Printf("💻 CPU模式 - 输入尺寸: %d\n", inputSize)
		return &YOLOConfig{
			InputSize:   inputSize,
			UseGPU:      false,
			LibraryPath: "",
		}
	}
}

// detectModelInputSize 从ONNX模型文件中检测输入尺寸
func detectModelInputSize(modelPath string) int {
	// 这是一个简化的实现，实际应该解析ONNX模型文件
	// 目前根据常见的YOLO模型文件名推断输入尺寸
	filename := filepath.Base(modelPath)
	filename = strings.ToLower(filename)
	
	// 常见的YOLO模型输入尺寸映射
	if strings.Contains(filename, "320") {
		return 320
	} else if strings.Contains(filename, "416") {
		return 416
	} else if strings.Contains(filename, "512") {
		return 512
	} else if strings.Contains(filename, "608") {
		return 608
	} else if strings.Contains(filename, "640") {
		return 640
	} else if strings.Contains(filename, "736") {
		return 736
	} else if strings.Contains(filename, "832") {
		return 832
	} else if strings.Contains(filename, "1024") {
		return 1024
	} else if strings.Contains(filename, "1280") {
		return 1280
	}
	
	// 根据模型类型推断
	if strings.Contains(filename, "yolo11n") || strings.Contains(filename, "yolo8n") {
		return 640 // nano模型通常使用640
	} else if strings.Contains(filename, "yolo11s") || strings.Contains(filename, "yolo8s") {
		return 640 // small模型通常使用640
	} else if strings.Contains(filename, "yolo11m") || strings.Contains(filename, "yolo8m") {
		return 640 // medium模型通常使用640
	} else if strings.Contains(filename, "yolo11l") || strings.Contains(filename, "yolo8l") {
		return 640 // large模型通常使用640
	} else if strings.Contains(filename, "yolo11x") || strings.Contains(filename, "yolo8x") {
		return 640 // xlarge模型通常使用640
	} else if strings.Contains(filename, "yolo12") {
		return 640 // YOLO12系列默认640
	}
	
	// 如果无法推断，返回0表示使用默认值
	return 0
}

// WithGPU 设置是否使用GPU
func (c *YOLOConfig) WithGPU(use bool) *YOLOConfig {
	c.UseGPU = use
	return c
}

// WithGPUDeviceID 设置GPU设备ID（仅在UseGPU=true时有效）
func (c *YOLOConfig) WithGPUDeviceID(deviceID int) *YOLOConfig {
	c.GPUDeviceID = deviceID
	return c
}

// WithLibraryPath 设置ONNX Runtime库路径
func (c *YOLOConfig) WithLibraryPath(path string) *YOLOConfig {
	c.LibraryPath = path
	return c
}

// DefaultDetectionOptions 默认检测选项（运行时级别）
func DefaultDetectionOptions() *DetectionOptions {
	return &DetectionOptions{
		ConfThreshold: 0.4,
		IOUThreshold:  0.5,
		DrawBoxes:     true,
		DrawLabels:    true,
		ShowFPS:       false,
		BoxColor:      "red",
		LabelColor:    "white",
		LineWidth:     2,
		FontSize:      12,
	}
}

// WithConfThreshold 设置置信度阈值
func (o *DetectionOptions) WithConfThreshold(threshold float32) *DetectionOptions {
	o.ConfThreshold = threshold
	return o
}

// WithIOUThreshold 设置IOU阈值
func (o *DetectionOptions) WithIOUThreshold(threshold float32) *DetectionOptions {
	o.IOUThreshold = threshold
	return o
}

// WithDrawBoxes 设置是否画框
func (o *DetectionOptions) WithDrawBoxes(draw bool) *DetectionOptions {
	o.DrawBoxes = draw
	return o
}

// WithDrawLabels 设置是否画标签
func (o *DetectionOptions) WithDrawLabels(draw bool) *DetectionOptions {
	o.DrawLabels = draw
	return o
}

// WithShowFPS 设置是否显示FPS
func (o *DetectionOptions) WithShowFPS(show bool) *DetectionOptions {
	o.ShowFPS = show
	return o
}

// WithBoxColor 设置框的颜色
func (o *DetectionOptions) WithBoxColor(color string) *DetectionOptions {
	o.BoxColor = color
	return o
}

// WithLabelColor 设置标签的颜色
func (o *DetectionOptions) WithLabelColor(color string) *DetectionOptions {
	o.LabelColor = color
	return o
}

// WithLineWidth 设置线条宽度
func (o *DetectionOptions) WithLineWidth(width int) *DetectionOptions {
	o.LineWidth = width
	return o
}

// WithFontSize 设置字体大小
func (o *DetectionOptions) WithFontSize(size int) *DetectionOptions {
	o.FontSize = size
	return o
}

// HighPerformanceConfig 高性能配置（自动检测并优化CPU/GPU）
// 注意：DefaultConfig现在已经是高性能配置，此函数保持向后兼容
func HighPerformanceConfig() *YOLOConfig {
	return DefaultConfig()
}

// MaxPerformanceGPUConfig GPU最大性能配置
func MaxPerformanceGPUConfig() *YOLOConfig {
	return &YOLOConfig{
		InputSize:   640, // GPU标准最大尺寸
		UseGPU:      true,
		LibraryPath: "",
	}
}

// MaxPerformanceCPUConfig CPU最大性能配置
func MaxPerformanceCPUConfig() *YOLOConfig {
	return &YOLOConfig{
		InputSize:   640, // 统一使用640输入尺寸以匹配模型要求
		UseGPU:      false,
		LibraryPath: "",
	}
}

// ExtremePerformanceConfig 极限性能配置（不计成本压榨硬件）
func ExtremePerformanceConfig() *YOLOConfig {
	if IsGPUAvailable() {
		fmt.Println("🔥 启用GPU极限压榨模式 - 不计成本！")
		return &YOLOConfig{
			InputSize:   640, // 使用标准尺寸确保稳定性
			UseGPU:      true,
			LibraryPath: "",
		}
	} else {
		fmt.Println("🔥 启用CPU极限压榨模式 - 不计成本！")
		return &YOLOConfig{
			InputSize:   640, // CPU也使用640确保准确性
			UseGPU:      false,
			LibraryPath: "",
		}
	}
}

// 预设配置（检测器级别）
var (
	// GPUConfig GPU配置
	GPUConfig = func() *YOLOConfig {
		return DefaultConfig().WithGPU(true)
	}

	// CPUConfig CPU配置
	CPUConfig = func() *YOLOConfig {
		return DefaultConfig().WithGPU(false)
	}
)
