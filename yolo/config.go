package yolo

// YOLOConfig YOLO检测器配置（检测器级别 - 创建时设置）
type YOLOConfig struct {
	InputSize   int    // 输入尺寸
	UseGPU      bool   // 是否使用GPU
	LibraryPath string // ONNX Runtime库路径
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

// DefaultConfig 返回默认配置（检测器级别）
func DefaultConfig() *YOLOConfig {
	return &YOLOConfig{
		InputSize:   640,
		UseGPU:      false,
		LibraryPath: "",
	}
}

// WithInputSize 设置输入尺寸
func (c *YOLOConfig) WithInputSize(size int) *YOLOConfig {
	c.InputSize = size
	return c
}

// WithGPU 设置是否使用GPU
func (c *YOLOConfig) WithGPU(use bool) *YOLOConfig {
	c.UseGPU = use
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
