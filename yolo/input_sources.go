package yolo

import (
	"fmt"
	"strings"
)

// InputSource 输入源类型
type InputSource struct {
	Type    string            // "file", "camera", "rtsp", "rtmp", "screen"
	Path    string            // 输入路径
	Options map[string]string // 额外选项
}

// NewFileInput 创建文件输入源
func NewFileInput(path string) *InputSource {
	return &InputSource{
		Type: "file",
		Path: path,
	}
}

// NewCameraInput 创建摄像头输入源
func NewCameraInput(device string) *InputSource {
	// 如果传入的是通用关键字，自动选择默认摄像头设备
	actualDevice := resolveCameraDevice(device)

	return &InputSource{
		Type: "camera",
		Path: actualDevice,
		Options: map[string]string{
			"f": "30", // 帧率
		},
	}
}

// resolveCameraDevice 解析摄像头设备，将通用关键字转换为具体设备
func resolveCameraDevice(device string) string {
	// 通用摄像头关键字
	cameraKeywords := []string{"camera", "cam", "webcam"}

	// 检查是否为通用关键字
	for _, keyword := range cameraKeywords {
		if strings.EqualFold(device, keyword) {
			// 返回默认摄像头设备
			return getDefaultCameraDevice()
		}
	}

	// 如果已经是具体设备路径，直接返回
	return device
}

// getDefaultCameraDevice 获取默认摄像头设备
func getDefaultCameraDevice() string {
	// 尝试检测可用的摄像头设备
	availableDevices := detectAvailableCameraDevices()

	if len(availableDevices) > 0 {
		return availableDevices[0] // 返回第一个可用的摄像头
	}

	// 如果没有检测到可用设备，返回默认值
	return "video=0" // Windows 默认
}

// detectAvailableCameraDevices 检测可用的摄像头设备
func detectAvailableCameraDevices() []string {
	var availableDevices []string

	// 常见的摄像头设备路径
	possibleDevices := []string{
		"video=0",     // Windows
		"video=1",     // Windows
		"/dev/video0", // Linux
		"/dev/video1", // Linux
		"0",           // 数字索引
		"1",           // 数字索引
	}

	// 这里可以添加实际的设备检测逻辑
	// 目前返回所有可能的设备，让 FFmpeg 去尝试
	for _, device := range possibleDevices {
		availableDevices = append(availableDevices, device)
	}

	return availableDevices
}

// GetCameraDeviceInfo 获取摄像头设备信息
func GetCameraDeviceInfo() map[string]string {
	devices := ListCameraDevices()
	deviceInfo := make(map[string]string)

	for i, device := range devices {
		deviceInfo[fmt.Sprintf("camera_%d", i)] = device
	}

	return deviceInfo
}

// NewRTSPInput 创建RTSP流输入源
func NewRTSPInput(url string) *InputSource {
	return &InputSource{
		Type: "rtsp",
		Path: url,
		Options: map[string]string{
			"rtsp_transport": "tcp",     // 使用TCP传输
			"stimeout":       "5000000", // 超时时间
		},
	}
}

// NewRTMPInput 创建RTMP流输入源
func NewRTMPInput(url string) *InputSource {
	return &InputSource{
		Type: "rtmp",
		Path: url,
	}
}

// NewScreenInput 创建屏幕录制输入源
func NewScreenInput() *InputSource {
	return &InputSource{
		Type: "screen",
		Path: "desktop", // Windows
		Options: map[string]string{
			"f": "30", // 帧率
		},
	}
}

// ListCameraDevices 列出可用的摄像头设备
func ListCameraDevices() []string {
	devices := []string{}

	// Windows系统
	devices = append(devices, "video=0") // 默认摄像头
	devices = append(devices, "video=1") // 第二个摄像头
	devices = append(devices, "video=2") // 第三个摄像头

	// Linux系统
	devices = append(devices, "/dev/video0")
	devices = append(devices, "/dev/video1")
	devices = append(devices, "/dev/video2")

	return devices
}

// ListScreenDevices 列出可用的屏幕设备
func ListScreenDevices() []string {
	devices := []string{}

	// Windows系统
	devices = append(devices, "desktop")   // 主屏幕
	devices = append(devices, "desktop=0") // 主屏幕
	devices = append(devices, "desktop=1") // 第二个屏幕

	// Linux系统
	devices = append(devices, ":0.0") // 主屏幕
	devices = append(devices, ":0.1") // 第二个屏幕

	return devices
}

// NewCameraInputWithDevice 创建指定设备的摄像头输入源
func NewCameraInputWithDevice(device string) *InputSource {
	return &InputSource{
		Type: "camera",
		Path: device,
		Options: map[string]string{
			"f": "30", // 帧率
		},
	}
}

// NewScreenInputWithDevice 创建指定设备的屏幕录制输入源
func NewScreenInputWithDevice(device string) *InputSource {
	return &InputSource{
		Type: "screen",
		Path: device,
		Options: map[string]string{
			"f": "30", // 帧率
		},
	}
}

// GetFFmpegInput 获取FFmpeg输入参数
func (is *InputSource) GetFFmpegInput() string {
	switch is.Type {
	case "file":
		return is.Path
	case "camera":
		return fmt.Sprintf("video=%s", is.Path)
	case "rtsp":
		return is.Path
	case "rtmp":
		return is.Path
	case "screen":
		return is.Path
	default:
		return is.Path
	}
}

// GetFFmpegOptions 获取FFmpeg选项
func (is *InputSource) GetFFmpegOptions() []string {
	var options []string

	for key, value := range is.Options {
		options = append(options, "-"+key, value)
	}

	return options
}

// IsRealTime 判断是否为实时输入源
func (is *InputSource) IsRealTime() bool {
	return is.Type != "file"
}

// Validate 验证输入源
func (is *InputSource) Validate() error {
	switch is.Type {
	case "file":
		if !strings.HasSuffix(strings.ToLower(is.Path), ".mp4") &&
			!strings.HasSuffix(strings.ToLower(is.Path), ".avi") &&
			!strings.HasSuffix(strings.ToLower(is.Path), ".mov") {
			return fmt.Errorf("不支持的文件格式: %s", is.Path)
		}
	case "camera":
		// 摄像头验证
		return nil
	case "rtsp", "rtmp":
		// 网络流验证
		return nil
	case "screen":
		// 屏幕录制验证
		return nil
	default:
		return fmt.Errorf("不支持的输入源类型: %s", is.Type)
	}

	return nil
}
