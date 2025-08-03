package yolo

import (
	"fmt"
	"net/url"
	"os/exec"
	"regexp"
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
			"f":           "dshow",     // Windows DirectShow
			"framerate":   "30",       // 帧率
			"video_size":  "640x480",  // 视频尺寸
			"pixel_format": "yuyv422", // 像素格式
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
// detectRealCameraDevices 通过FFmpeg实际检测可用的摄像头设备
func detectRealCameraDevices() []string {
	var devices []string

	// 执行FFmpeg命令检测设备
	cmd := exec.Command("ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// FFmpeg命令失败，返回空列表
		return devices
	}

	// 解析输出，提取视频设备名称
	outputStr := string(output)
	lines := strings.Split(outputStr, "\n")

	// 使用正则表达式匹配视频设备
	videoDeviceRegex := regexp.MustCompile(`\[dshow @ [^\]]+\] "([^"]+)" \(video\)`)

	for _, line := range lines {
		matches := videoDeviceRegex.FindStringSubmatch(line)
		if len(matches) > 1 {
			deviceName := matches[1]
			// 过滤掉虚拟摄像头
			if !strings.Contains(strings.ToLower(deviceName), "virtual") {
				devices = append(devices, deviceName)
			}
		}
	}

	return devices
}

func detectAvailableCameraDevices() []string {
	var availableDevices []string

	// 尝试通过FFmpeg检测实际的摄像头设备
	realDevices := detectRealCameraDevices()
	if len(realDevices) > 0 {
		return realDevices
	}

	// 如果检测失败，返回常见的摄像头设备路径
	possibleDevices := []string{
		"USB2.0 HD UVC WebCam", // 实际检测到的设备名
		"video=0",              // Windows
		"video=1",              // Windows
		"/dev/video0",          // Linux
		"/dev/video1",          // Linux
		"0",                    // 数字索引
		"1",                    // 数字索引
	}

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
// 支持带认证的URL格式: rtsp://username:password@host:port/path
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
// 支持带认证的URL格式: rtmp://username:password@host:port/app/stream
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
		// 检查是否已经包含video=前缀，避免重复添加
		if strings.HasPrefix(is.Path, "video=") {
			return is.Path
		}
		// 检查是否为Linux设备路径
		if strings.HasPrefix(is.Path, "/dev/video") {
			return is.Path
		}
		// 对于纯数字索引，添加video=前缀
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
	return is.Type == "camera" || is.Type == "rtsp" || is.Type == "rtmp" || is.Type == "screen"
}

// GetInputType 获取输入源类型
func (is *InputSource) GetInputType() string {
	return is.Type
}

// GetCameraIndex 获取摄像头索引（用于兼容性）
func (is *InputSource) GetCameraIndex() int {
	if is.Type != "camera" {
		return -1
	}
	
	// 尝试从路径中提取数字索引
	if len(is.Path) == 1 && is.Path >= "0" && is.Path <= "9" {
		return int(is.Path[0] - '0')
	}
	
	// 从 video=X 格式中提取
	if strings.HasPrefix(is.Path, "video=") && len(is.Path) > 6 {
		if is.Path[6] >= '0' && is.Path[6] <= '9' {
			return int(is.Path[6] - '0')
		}
	}
	
	return 0 // 默认返回0
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
		if is.Path == "" {
			return fmt.Errorf("摄像头设备路径不能为空")
		}
		// 检查常见的摄像头设备格式
		validFormats := []string{"video=", "/dev/video", "0", "1", "2", "3", "4"}
		isValid := false
		
		// 检查是否为标准格式
		for _, format := range validFormats {
			if strings.HasPrefix(is.Path, format) || is.Path == format {
				isValid = true
				break
			}
		}
		
		// 如果不是标准格式，检查是否为完整的设备名称（包含字母和空格）
		if !isValid {
			// 允许包含字母、数字、空格、点号、括号的设备名称
			deviceNameRegex := regexp.MustCompile(`^[a-zA-Z0-9\s\.\(\)_-]+$`)
			if deviceNameRegex.MatchString(is.Path) {
				isValid = true
			}
		}
		
		if !isValid {
			return fmt.Errorf("不支持的摄像头设备格式: %s，支持的格式: video=0, /dev/video0, 0, 1, 2等或完整设备名称", is.Path)
		}
		return nil
	case "rtsp", "rtmp":
		// 网络流验证
		if is.Path == "" {
			return fmt.Errorf("网络流URL不能为空")
		}
		
		// 检查URL格式
		if !strings.HasPrefix(is.Path, "rtsp://") && !strings.HasPrefix(is.Path, "rtmp://") {
			return fmt.Errorf("无效的流URL格式: %s，必须以 rtsp:// 或 rtmp:// 开头", is.Path)
		}
		
		// 验证URL格式的有效性
		parsedURL, err := url.Parse(is.Path)
		if err != nil {
			return fmt.Errorf("无效的URL格式: %s，错误: %v", is.Path, err)
		}
		
		// 检查主机名是否存在（去除认证信息后）
		hostname := parsedURL.Hostname()
		if hostname == "" {
			return fmt.Errorf("URL缺少主机名: %s", is.Path)
		}
		
		// 检查是否包含认证信息
		if parsedURL.User != nil {
			username := parsedURL.User.Username()
			_, hasPassword := parsedURL.User.Password()
			if username != "" {
				fmt.Printf("检测到认证信息 - 用户名: %s, 密码: %s\n", username, 
					func() string {
						if hasPassword {
							return "[已设置]"
						}
						return "[未设置]"
					}())
			}
		}
		
		// 检查协议是否匹配
		if is.Type == "rtsp" && parsedURL.Scheme != "rtsp" {
			return fmt.Errorf("RTSP输入源必须使用rtsp://协议: %s", is.Path)
		}
		if is.Type == "rtmp" && parsedURL.Scheme != "rtmp" {
			return fmt.Errorf("RTMP输入源必须使用rtmp://协议: %s", is.Path)
		}
		
		return nil
	case "screen":
		// 屏幕录制验证
		return nil
	default:
		return fmt.Errorf("不支持的输入源类型: %s", is.Type)
	}

	return nil
}
